import logging
from typing import List, Dict

from GenDataset.single.workload.workload import Label, Query
from GenDataset.single.estimator.estimator import Oracle
from GenDataset.single.dataset.dataset import Table

L = logging.getLogger(__name__)


def generate_labels_for_queries(table: Table, queryset: Dict[str, List[Query]]) -> Dict[str, List[Label]]:
    oracle = Oracle(table)
    labels = {}
    for group, queries in queryset.items():
        l = []
        for i, q in enumerate(queries):
            card, _ = oracle.query(q)
            l.append(Label(cardinality=card, selectivity=card / table.row_num))
            if (i + 1) % 1000 == 0:
                L.info(f"{i + 1} labels generated for {group}")
        labels[group] = l

    return labels


def generate_label_for_query(table: Table, query: Query):
    oracle = Oracle(table)
    card, _ = oracle.query2(query)
    return card


def generate_label_by_standard_sample(std_q_error, sample_q_error):
    if sample_q_error <= std_q_error:
        return 1
    else:
        return 0
