# Copyright 2019 The Google Research Authors.
# Copyright [2024] Stefan Dendorfer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
This python script generates run data using the NAS-Bench 101 tabular dataset. 
It implements a regularized evolution algorithm for a fixed time budget and 
saves the maximum accuracy achieved to a json. The two experiments respectively 
consists of 30 and 1000 runs.

Requirements:

-This script requires the nasbench package to be installed. A tensorflow2 version
relying on version 2.15.0 can be installed via 
    pip install -i https://test.pypi.org/simple/ nasbench-TF2

-This script required the tabular benchmark to be downloaded. Instructions can
be found here: 
https://github.com/google-research/nasbench?tab=readme-ov-file#download-the-dataset
please make sure to adjust the NB_DATASET_PATH accordingly. 

'''


import random
import copy
import os 
import numpy as np
import tensorflow as tf
import json
from nasbench.api import NASBench, ModelSpec

NB_DATASET_PATH = os.path.join("data", "nasbench_only108.tfrecord")

# Useful constants for nasbench
INPUT = "input"
OUTPUT = "output"
CONV3X3 = "conv3x3-bn-relu"
CONV1X1 = "conv1x1-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2  # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2  # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]  # Binary adjacency matrix

physical_devices = tf.config.experimental.list_physical_devices("GPU")

if len(physical_devices) > 0:
    print("Using GPU for tf.")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU.")



def random_spec(nasbench: NASBench):
    """Returns a random valid spec."""
    while True:
        matrix = np.random.choice(ALLOWED_EDGES,
                                  size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(ALLOWED_OPS, size=NUM_VERTICES).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT
        spec = ModelSpec(matrix=matrix, ops=ops)
        if nasbench.is_valid(spec):
            return spec


def mutate_spec(old_spec, nasbench: NASBench, mutation_rate=1.0):
    """Computes a valid mutated spec from the old_spec."""
    while True:
        new_matrix = copy.deepcopy(old_spec.original_matrix)
        new_ops = copy.deepcopy(old_spec.original_ops)

        # In expectation, V edges flipped (note that most end up being pruned).
        edge_mutation_prob = mutation_rate / NUM_VERTICES
        for src in range(0, NUM_VERTICES - 1):
            for dst in range(src + 1, NUM_VERTICES):
                if random.random() < edge_mutation_prob:
                    new_matrix[src, dst] = 1 - new_matrix[src, dst]

        # In expectation, one op is resampled.
        op_mutation_prob = mutation_rate / OP_SPOTS
        for ind in range(1, NUM_VERTICES - 1):
            if random.random() < op_mutation_prob:
                available = [
                    o for o in nasbench.config["available_ops"]
                    if o != new_ops[ind]
                ]
                new_ops[ind] = random.choice(available)

        new_spec = ModelSpec(new_matrix, new_ops)
        if nasbench.is_valid(new_spec):
            return new_spec
        

def random_combination(iterable, sample_size):
    """Random selection from itertools.combinations(iterable, r)."""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)

def run_revolution_search(
    nasbench: NASBench,
    max_time_budget=5e6,
    population_size=50,
    tournament_size=10,
    mutation_rate=0.5
):
    """Run a single roll-out of regularized evolution to a fixed time budget."""

    times, best_valids, best_tests = [0.0], [0.0], [0.0]
    population = []  # (validation, spec) tuples

    # For the first population_size individuals, seed the population with
    # randomly generated cells.
    for _ in range(population_size):
        spec = random_spec(nasbench)
        data = nasbench.query(spec)
        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)
        population.append((data["validation_accuracy"], spec))

        if data["validation_accuracy"] > best_valids[-1]:
            best_valids.append(data["validation_accuracy"])
            best_tests.append(data["test_accuracy"])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        if time_spent > max_time_budget:
            break
    # After the population is seeded, proceed with evolving the population.
    while True:
        sample = random_combination(population, tournament_size)
        best_spec = sorted(sample, key=lambda i: i[0])[-1][1]
        new_spec = mutate_spec(best_spec, nasbench, mutation_rate)

        data = nasbench.query(new_spec)
        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)

        # In regularized evolution, we kill the oldest individual.
        population.append((data["validation_accuracy"], new_spec))
        population.pop(0)

        if data["validation_accuracy"] > best_valids[-1]:
            best_valids.append(data["validation_accuracy"])
            best_tests.append(data["test_accuracy"])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        if time_spent > max_time_budget:
            break

    return times, best_valids, best_tests


if __name__ == "__main__":

    nasb = NASBench(NB_DATASET_PATH)
    valids_30 = []
    tests_30 = []
    valids_1000 = []
    tests_1000 = []

    budget = int(1e6)
    n_30 = 30
    n_1000 = 1000
    for repeat in range(n_30):
        nasb.reset_budget_counters()
        _, best_valid, best_test = run_revolution_search(nasb, budget, 32)
        valids_30.append(best_valid[-1])
        tests_30.append(best_test[-1])
        
    print(f"{n_30} runs done")
    for repeat in range(n_1000):
        if (repeat % 100 == 0):
            print('Running repeat %d' % (repeat))

        nasb.reset_budget_counters()
        _, best_valid, best_test = run_revolution_search(nasb, budget, 32)
        valids_1000.append(best_valid[-1])
        tests_1000.append(best_test[-1])

    run_data = {}
    run_data['valids_30'] = valids_30
    run_data['valids_1000'] = valids_1000
    run_data['tests_30'] = tests_30
    run_data['tests_1000'] = tests_1000

    # Specify the file path where you want to save the JSON file
    file_path = "./data/generated_run_data.json"

    # Save the run_data dictionary as JSON
    with open(file_path, 'w') as f:
        json.dump(run_data, f)

    print("Run data saved successfully.")
        

