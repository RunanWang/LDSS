import time
import numpy as np
from typing import Tuple, Any
from GenDataset.single.workload.workload import Query, query_2_triple
from GenDataset.single.dataset.dataset import Table
from utils.log import Log

L = Log(__name__).get_logger()


class Estimator(object):
    """Base class for a cardinality estimator."""

    def __init__(self, table: Table, **kwargs: Any) -> None:
        self.table = table
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
    def __init__(self, table):
        super(Oracle, self).__init__(table=table)

    def query(self, query):
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        start_stmp = time.time()
        bitmap = np.ones(self.table.row_num, dtype=bool)
        for c, o, v in zip(columns, operators, values):
            bitmap &= OPS[o](self.table.data[c], v)
        card = bitmap.sum()
        dur_ms = (time.time() - start_stmp) * 1e3
        return card, dur_ms

    def query2(self, query):
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        start_stmp = time.time()
        bitmap = np.ones(self.table.row_num, dtype=bool)
        df = self.table.data
        for c, o, v in zip(columns, operators, values):
            bitmap = OPS[o](df[c], v)
            df = df[bitmap]
            bitmap = bitmap[bitmap]
        card = bitmap.sum()
        dur_ms = (time.time() - start_stmp) * 1e3
        return card, dur_ms

    def like(self, like_str, str_col):
        qualified = 0
        for index, row in self.table.data.iterrows():
            if like_str in str(row[str_col]):
                qualified += 1
        return qualified
