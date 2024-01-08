# LDSS

PyTorch implementation of Learned Dynamic Sample Selection (LDSS) to estimate the cardinality of queries.

## Requirements

conda env create -f environment.yaml

## Usage

### Single table

Download Dataset and put them in /data folder.

Following we use census as example.

```shell

    python main.py --phase genload --table single --dataset census --workload generate

    python main.py --phase gendata --table single --dataset census --workload generate

    python main.py --phase train --table single --dataset census --workload generate

    python main.py --phase test --table single --dataset census --workload generate --testW test
    
```

Results are shown in /output/log/model-test-census.log


### JOB-light

```shell

    cd data

    sh get_job.sh

    cd ..

    python main.py --phase gendata --table multi --dataset job-light --workload train

    python main.py --phase gendata --table multi --dataset job-light --workload job-light

    python main.py --phase train --table multi --dataset job-light --workload train

    python main.py --phase test --table multi --dataset job-light --workload train --testW job-light

```

Results are shown in /output/log/model-test-job-light.log



### New Dataset

#### Data

Each folder in /data path is a dataset.

Each table in a dataset should be saved in a csv file format.

Examples can be found at /data/README.md.

#### Query Workload

We can put all SQLs of a workload into a .sql file in /job-light folder.

Examples can be found at /workload/README.md.

#### Schema

New Dataset should contain a TABLE_COL schema.

For Multi-table, dataset should contain a JOIN_COL schema.

All schemas can be found at constants.py

# Cite

```bibtex
    @article{LDSS,
        title = {Cardinality estimation via learned dynamic sample selection},
        journal = {Information Systems},
        volume = {117},
        pages = {102252},
        year = {2023},
        issn = {0306-4379},
        doi = {https://doi.org/10.1016/j.is.2023.102252},
        url = {https://www.sciencedirect.com/science/article/pii/S0306437923000881},
        author = {Run-An Wang and Zhaonian Zou and Ziqi Jing}
    }
```
