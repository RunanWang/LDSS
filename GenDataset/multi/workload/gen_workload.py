import random
import numpy as np
from typing import Dict, Any
import copy

import GenDataset.multi.workload.generator as generator
from GenDataset.multi.workload.generator import QueryGenerator
from GenDataset.multi.workload.workload import dump_queryset, sql_2_query
from GenDataset.multi.dataset.table import Database
from utils.log import Log

L = Log(__name__).get_logger()


def get_focused_table(table, ref_table, win_ratio):
    focused_table = copy.deepcopy(table)
    win_size = int(win_ratio * len(ref_table.data))
    focused_table.data = focused_table.data.tail(win_size).reset_index(drop=True)
    focused_table.parse_columns()
    return focused_table


def generate_workload(seed: int, name: str, params: Dict[str, Dict[str, Any]]):
    random.seed(seed)
    np.random.seed(seed)

    table_funcs = {getattr(generator, f"tsf_{a}"): v for a, v in params['table'].items()}
    attr_funcs = {getattr(generator, f"asf_{a}"): v for a, v in params['attr'].items()}
    center_funcs = {getattr(generator, f"csf_{c}"): v for c, v in params['center'].items()}
    width_funcs = {getattr(generator, f"wsf_{w}"): v for w, v in params['width'].items()}

    L.info("Loading database...")
    db = Database()
    gen = QueryGenerator(
        database=db,
        table=table_funcs,
        attr=attr_funcs,
        center=center_funcs,
        width=width_funcs,
        table_params=params.get('table_params') or {},
        attr_params=params.get('attr_params') or {},
        center_params=params.get('center_params') or {},
        width_params=params.get('width_params') or {},
    )
    queryset = {}
    for group, num in params['number'].items():
        L.info(f"Start generate workload with {num} queries for {group}...")
        queries = []
        for i in range(num):
            queries.append(gen.generate())
            if (i + 1) % 1000 == 0:
                L.info(f"{i + 1} queries generated")
        queryset[group] = queries
    L.info("Saving queryset to disk...")
    dump_queryset(name, queryset)
    # dump_sql(name, "test")
    return queryset


def generate_workload_from_sql(sql_path, database):
    queries = []
    for sql in open(sql_path):
        queries.append(sql_2_query(sql, database))
    return queries
