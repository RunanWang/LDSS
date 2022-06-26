import random
from typing import Dict, List, Any, Optional, Tuple
from typing_extensions import Protocol

import numpy as np
import pandas as pd

from GenDataset.multi.dataset.table import Database, Column
from GenDataset.multi.workload.workload import Query, new_query
from utils.log import Log

L = Log(__name__).get_logger()

"""====== Table Selection Functions ======"""


class TableSelFunc(Protocol):
    def __call__(self, database: Database, params: Dict[str, Any]) -> List[str]: ...


def tsf_pred_number(database: Database, params: Dict[str, Any]) -> List[str]:
    if 'whitelist' in params:
        table_domain = params['whitelist']
    else:
        blacklist = params.get('blacklist') or []
        table_domain = [t for t in list(database.table.keys()) if t not in blacklist]
    nums = params.get('nums')
    nums = nums or np.random.choice(range(1, len(table_domain) + 1))
    if 'must_list' in params:
        must_list = params.get('must_list')
        table_domain = [t for t in list(database.table.keys()) if t not in must_list]
        nums = nums - len(must_list)
        ans = np.random.choice(table_domain, size=nums, replace=False).tolist()
        ans = must_list + ans
        ans = np.array(ans)
    else:
        ans = np.random.choice(table_domain, size=nums, replace=False)
    return ans


"""====== Attribute Selection Functions ======"""


class AttributeSelFunc(Protocol):
    def __call__(self, database: Database, tables: List[str], params: Dict[str, Any]) -> Dict[str, List[str]]: ...


def asf_pred_number(database: Database, tables: List[str], params: Dict[str, Any]) -> Dict[str, List[str]]:
    attrs = {}
    for t in tables:
        attr_domain = [a for a in list(database.table[t].data.columns) if a != database.join_col[t] and a != "id"]
        max_nums = params.get('max_nums')
        min_nums = params.get('min_nums')
        nums = np.random.choice(range(min_nums, max_nums))
        nums = min(nums, len(attr_domain))
        attrs[t] = np.random.choice(attr_domain, size=nums, replace=False)
    return attrs


"""====== Center Selection Functions ======"""


class CenterSelFunc(Protocol):
    def __call__(self, database: Database, attrs: Dict[str, List[str]], params: Dict[str, Any]) -> Dict[
        str, List[Any]]: ...


ROW_CACHE = None
GLOBAL_COUNTER = 1000


def csf_distribution(database: Database, attrs: Dict[str, List[str]], params: Dict[str, Any]) -> Dict[str, List[Any]]:
    global GLOBAL_COUNTER
    global ROW_CACHE
    BASE_TABLE = database.base_table
    if GLOBAL_COUNTER >= 1000:
        data_from = params.get('data_from') or 0
        ROW_CACHE = np.random.choice(range(data_from, len(database.table[BASE_TABLE].data)), size=1000)
        GLOBAL_COUNTER = 0
    centers = {}
    base_id = database.table[BASE_TABLE].data.at[ROW_CACHE[GLOBAL_COUNTER], database.join_col[BASE_TABLE]]
    for t in attrs.keys():
        if t == BASE_TABLE:
            selected_t = database.table[t].data
        else:
            selected_t = database.table[t].data[database.table[t].data[database.join_col[t]] == base_id]
            selected_t.reset_index(inplace=True)
        if selected_t is None or len(selected_t) == 0:
            selected_t = database.table[t].data
        index = np.random.choice(range(0, len(selected_t)))
        centers[t] = [selected_t.at[index, a] for a in attrs[t]]
    GLOBAL_COUNTER += 1
    return centers


def csf_vocab_ood(database: Database, attrs: Dict[str, List[str]], params: Dict[str, Any]) -> Dict[str, List[Any]]:
    centers = {}
    for t in attrs.keys():
        centers[t] = []
        for a in attrs[t]:
            col = database.table[t].columns[a]
            centers[t].append(np.random.choice(col.vocab))
    return centers


"""====== Width Selection Functions ======"""


class WidthSelFunc(Protocol):
    def __call__(self, database: Database, attrs: Dict[str, List[str]], centers: Dict[str, List[Any]],
                 params: Dict[str, Any]) -> Query: ...


def parse_range(col: Column, left: Any, right: Any) -> Optional[Tuple[str, Any]]:
    if left <= col.min_val:
        return '<=', round(right)
    if right >= col.max_val:
        return '>=', round(left)
    if round(left) == round(right):
        return '=', round(left)
    return '=', round((left + right) / 2)


def wsf_uniform(database: Database, attrs: Dict[str, List[str]], centers: Dict[str, List[Any]],
                params: Dict[str, Any]) -> Query:
    query = new_query(database, list(attrs.keys()))
    for t in attrs.keys():
        for a, c in zip(attrs[t], centers[t]):
            # NaN/NaT literal can only be assigned to = operator
            if pd.isnull(c):
                continue
            col = database.table[t].columns[a]
            width = random.uniform(0, col.max_val - col.min_val)
            query.predicates[t][a] = parse_range(col, c - width / 2, c + width / 2)
    return query


def wsf_exponential(database: Database, attrs: Dict[str, List[str]], centers: Dict[str, List[Any]],
                    params: Dict[str, Any]) -> Query:
    query = new_query(database, list(attrs.keys()))
    for t in attrs.keys():
        for a, c in zip(attrs[t], centers[t]):
            # NaN/NaT literal can only be assigned to = operator
            if pd.isnull(c):
                continue
            col = database.table[t].columns[a]
            lmd = 1 / ((col.max_val - col.min_val) / 10)
            width = random.expovariate(lmd)
            query.predicates[t][a] = parse_range(col, c - width / 2, c + width / 2)
    return query


class QueryGenerator(object):
    database: Database
    table: Dict[TableSelFunc, float]
    attr: Dict[AttributeSelFunc, float]
    center: Dict[CenterSelFunc, float]
    width: Dict[WidthSelFunc, float]
    table_params: Dict[str, Any]
    attr_params: Dict[str, Any]
    center_params: Dict[str, Any]
    width_params: Dict[str, Any]

    def __init__(
            self, database: Database,
            table: Dict[TableSelFunc, float],
            attr: Dict[AttributeSelFunc, float],
            center: Dict[CenterSelFunc, float],
            width: Dict[WidthSelFunc, float],
            table_params: Dict[str, Any],
            attr_params: Dict[str, Any],
            center_params: Dict[str, Any],
            width_params: Dict[str, Any]
    ) -> None:
        self.database = database
        self.table = table
        self.attr = attr
        self.center = center
        self.width = width
        self.table_params = table_params
        self.attr_params = attr_params
        self.center_params = center_params
        self.width_params = width_params

    def generate(self) -> Query:
        table_func = np.random.choice(list(self.table.keys()), p=list(self.table.values()))
        table_lst = table_func(self.database, self.table_params)

        attr_func = np.random.choice(list(self.attr.keys()), p=list(self.attr.values()))
        attr_lst = attr_func(self.database, table_lst, self.attr_params)

        center_func = np.random.choice(list(self.center.keys()), p=list(self.center.values()))
        center_lst = center_func(self.database, attr_lst, self.center_params)

        width_func = np.random.choice(list(self.width.keys()), p=list(self.width.values()))
        return width_func(self.database, attr_lst, center_lst, self.width_params)
