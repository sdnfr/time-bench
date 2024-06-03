# time-bench

This repository contains the code implementation of the paper titled "Time Is All You Need". The provided code allows you to reproduce the charts and work with plug and play testing using data from the publication, eliminating the need to rerun any NAS (Neural Architecture Search) runs.

## Quick Start

To quickly explore the data from the paper without downloading tabular datasets and running simulations, you can utilize the provided notebooks.

```bash
git clone <repository_url>
pip install -r requirements.txt
```

## Extensive Setup

To generate your own data using NAS Bench 101 and NATS-Bench, you need to download and install the required tabular benchmarks.

### NAS Bench 101 Setup

1. Download the `.tfrecord` file from Google Drive to the `./data` folder. Follow the instructions [here](https://github.com/google-research/nasbench?tab=readme-ov-file#download-the-dataset) to download either the full version or the smaller version of the dataset.

2. After downloading, the directory structure should resemble the following:

```
./
+--data/
    |-- nasbench_full.tfrecord
    |-- nasbench_only108.tfrecord
```

3. Since there is no maintained NAS Bench 101 repository that supports TensorFlow 2, the `nasbench-TF2` package is used. It's automatically installed via:

```bash
pip install -i https://test.pypi.org/simple/ nasbench-TF2
```

Note: TensorFlow 2.15.0 is required to support NAS Bench.

### NATS-Bench Setup

1. Install the `nats_bench` package for evaluation:

```bash
pip install nats_bench
```

2. Download the dataset into the `$TORCH_HOME` directory. Instructions can be found [here](https://github.com/D-X-Y/AutoDL-Projects?tab=readme-ov-file#requirements-and-preparation).

3. Additionally, to run algorithms such as REINFORCE or regularized evolution, you need the AUTODL repository. Install it as follows:

```bash
mkdir thirdparty
cd thirdparty
git clone https://github.com/D-X-Y/AutoDL-Projects.git autodl
```

## Usage

The notebooks in `timebench` demonstrate the evaluations performed to produce the output of the paper.

- `n1`: Utilizes data from the LayerNAS paper (https://arxiv.org/abs/2304.11517) to output probabilities.
- `n2`: Utilizes data from two experiments with 30 and 1000 runs, and outputs histograms and normal distributions of succeeding experiments applying the central limit theorem. You can use NAS Bench 101 to produce your own run data via the `create_run_data.py` script in the `scripts` folder.
- `n3`: Plots the application of the central limit theorem using data from the NATS-Bench paper (https://arxiv.org/abs/2009.00437).
- `n4`: Shows the extension of time budget for the NATS-Bench when comparing REINFORCE and regularized evolution. The data of a run is provided. You can generate your own run data using NATS-Bench via the `create_nats_data.py` script in the `scripts` folder.

Feel free to explore and reproduce the results using the provided code and datasets. If you encounter any issues or have questions, please refer to the respective paper or raise an issue in this repository.
