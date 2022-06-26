# How to format a dataset

Each folder in /data path is a dataset.

Each table in a dataset shold be saved in a csv file format.

For single table, the csv file in its folder is its dataset.

For multi-table, the csv files in its folder are its dataset.

## Single Table Example

**The Dataset comes from https://arxiv.org/abs/2012.06743 .**

The four real-world datasets can be downloaded from [here](https://www.dropbox.com/s/5bmvc1si5hysapf/data.tar.gz?dl=0).

Here we contain census dataset as an example.

## Multi-table Example

We use JOB-light dataset as an example.

Get JOB dataset from [here](http://homepages.cwi.nl/~boncz/job/imdb.tgz).

Process JOB into JOB-light

```shell
    sh get_job.sh
```