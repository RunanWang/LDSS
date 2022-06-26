import time
import numpy as np
import pandas as pd
from typing import Tuple, Any
from GenDataset.multi.workload.workload import Query, query_2_triple
from GenDataset.multi.dataset.table import Database
from utils.log import Log

L = Log(__name__).get_logger()


class Estimator(object):
    """Base class for a cardinality estimator."""

    def __init__(self, database: Database, **kwargs: Any) -> None:
        self.database = database
        self.params = dict(kwargs)

    def __repr__(self) -> str:
        pstr = ';'.join([f"{p}={v}" for p, v in self.params.items()])
        return f"{self.__class__.__name__.lower()}-{pstr}"

    def query(self, query: Query) -> Tuple[float, float]:
        """return est_card, dur_ms"""
        raise NotImplementedError


def in_between(data: Any, val: Tuple[Any, Any]) -> bool:
    assert len(val) == 2
    lrange, rrange = val
    return np.greater_equal(data, lrange) & np.less_equal(data, rrange)


OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '[]': in_between
}


class Oracle(Estimator):
    def __init__(self, database: Database):
        self.database = database

    def query(self, query):
        start = time.time()
        part_card = None
        for table_name, join_column in query.join:
            data_df = self.database.table[table_name].data
            columns, operators, values = query_2_triple(query, table_name, with_none=False, split_range=False)
            row_num = len(data_df)
            bitmap = np.ones(row_num, dtype=bool)
            for c, o, v in zip(columns, operators, values):
                bitmap &= OPS[o](data_df[c], v)
            card = pd.DataFrame(data_df[bitmap].groupby(by=self.database.join_col[table_name]).size(), columns=["size"])
            if part_card is None:
                part_card = card
            else:
                join_df = card.join(part_card, lsuffix='_left', rsuffix='_right')
                join_df = join_df.fillna(0)
                result_temp = pd.DataFrame(join_df["size_left"] * join_df['size_right'], columns=["size"])
                part_card = result_temp.drop(result_temp[result_temp["size"] == 0].index)
        final_card = part_card.sum()["size"]
        dur_ms = (time.time() - start) * 1e3
        return final_card, dur_ms
