import logging

from GenDataset.multi.workload.workload import Query
from GenDataset.multi.estimator.estimator import Oracle
from GenDataset.multi.dataset.table import Database

L = logging.getLogger(__name__)


def generate_label_for_query(database: Database, query: Query):
    oracle = Oracle(database)
    card, _ = oracle.query(query)
    return card


def generate_label_by_standard_sample(std_q_error, sample_q_error):
    if sample_q_error <= std_q_error:
        return 1
    else:
        return 0
