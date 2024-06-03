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
This python script generates run data using the NATS-Bench tabular dataset. 
It runs two experiments with the reinforce algorithm and regularized evolution.

Requirements:

- 
-This script requires auto-dl projects to be installed as thirdparty

    mkdir thirdparty
    cd thirdparty
    git clone https://github.com/D-X-Y/AutoDL-Projects.git autodl


-This script required the tabular benchmark to be downloaded. Instructions can
be found here: 
https://github.com/D-X-Y/AutoDL-Projects?tab=readme-ov-file#requirements-and-preparation
please make sure to add this into the torch home directory. 
'''

import subprocess


if __name__ == "__main__":
    BUDGET = 200000
    LOOPS = 1

    command = [
        "python", 
        "./thirdparty/autodl/exps/NATS-algos/regularized_ea.py",
        "--save_dir", "./data/generated", 
        "--dataset", "cifar10",
        "--search_space", "tss",
        "--time_budget", str(BUDGET),
        "--loops_if_rand", str(LOOPS),
        "--ea_cycles", "200",
        "--ea_population", "20",
        "--ea_sample_size", "10"
    ]
    print("Running regularized evolution...")
    subprocess.run(command, capture_output=True, text=True)
    print("done")

    command = [
        "python", 
        "./thirdparty/autodl/exps/NATS-algos/reinforce.py",
        "--save_dir", "./data/generated", 
        "--dataset", "cifar10",
        "--search_space", "tss",
        "--time_budget", str(BUDGET),
        "--loops_if_rand", str(LOOPS),
        "--learning_rate", str(0.01),
    ]
    print("Running reinforce...")
    subprocess.run(command, capture_output=True, text=True)
    print("done")
