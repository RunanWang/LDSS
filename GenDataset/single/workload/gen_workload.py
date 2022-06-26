import random
import logging
import numpy as np
from typing import Dict, Any
import copy

import GenDataset.single.workload.generator as generator
from GenDataset.single.workload.generator import QueryGenerator
from GenDataset.single.workload.workload import dump_gen_queryset
from GenDataset.single.dataset.dataset import load_table

L = logging.getLogger(__name__)


def get_focused_table(table, ref_table, win_ratio):
    focused_table = copy.deepcopy(table)
    win_size = int(win_ratio * len(ref_table.data))
    focused_table.data = focused_table.data.tail(win_size).reset_index(drop=True)
    focused_table.parse_columns()
    return focused_table


def generate_workload(seed: int, dataset: str, params: Dict[str, Dict[str, Any]]):
    random.seed(seed)
    np.random.seed(seed)

    attr_funcs = {getattr(generator, f"asf_{a}"): v for a, v in params['attr'].items()}
    center_funcs = {getattr(generator, f"csf_{c}"): v for c, v in params['center'].items()}
    width_funcs = {getattr(generator, f"wsf_{w}"): v for w, v in params['width'].items()}

    L.info("Load table...")
    table = load_table(dataset)

    qgen = QueryGenerator(
        table=table,
        attr=attr_funcs,
        center=center_funcs,
        width=width_funcs,
        attr_params=params.get('attr_params') or {},
        center_params=params.get('center_params') or {},
        width_params=params.get('width_params') or {})

    queryset = {}
    for group, num in params['number'].items():
        L.info(f"Start generate workload with {num} queries for {group}...")
        queries = []
        for i in range(num):
            queries.append(qgen.generate())
            if (i + 1) % 1000 == 0:
                L.info(f"{i + 1} queries generated")
        queryset[group] = queries

    L.info("Dump queryset to disk...")
    dump_gen_queryset(dataset, queryset)
    return queryset
