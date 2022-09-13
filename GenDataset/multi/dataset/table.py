import copy
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd

from constants import TEMP_ROOT, PKL_PROTO
from utils.log import Log

L = Log(__name__).get_logger()


class Database(object):
    def __init__(self, data_root, table_list, join_col, base_table):
        self.table = {}
        self.data = {}
        self.join_col = join_col
        self.table_list = table_list
        self.base_table = base_table
        for t in table_list:
            L.info(f"Loading table {t}")
            self.table[t] = load_table(t, data_root)
        self.join_col = join_col


class Column(object):
    def __init__(self, name, data):
        self.name = name
        self.type = data.dtype

        # parse vocabulary
        self.vocab, self.has_nan = self.__parse_vocab(data)
        self.vocab_size = len(self.vocab)
        if self.has_nan and self.vocab_size > 1:
            self.min_val = self.vocab[1]
        else:
            self.min_val = self.vocab[0]
        self.max_val = self.vocab[-1]

    def __repr__(self):
        return f'Column({self.name}, type={self.type}, vocab size={self.vocab_size}, min={self.min_val}, max={self.max_val}, has NaN={self.has_nan})'

    def __parse_vocab(self, data):
        # pd.isnull returns true for both np.nan and np.datetime64('NaT').
        is_nan = pd.isnull(data)
        contains_nan = np.any(is_nan)
        # NOTE: np.sort puts NaT values at beginning, and NaN values at end.
        # For our purposes we always add any null value to the beginning.
        vs = np.sort(np.unique(data[~is_nan]))
        if contains_nan:
            vs = np.insert(vs, 0, np.nan)
        return vs, contains_nan

    def normalize(self, data):
        """Normalize data to range [0, 1]"""
        data = np.array(data, dtype=np.float32)
        if self.min_val >= self.max_val:
            L.warning(f"column {self.name} has min value {self.min_val} >= max value{self.max_val}")
            return np.zeros(len(data)).astype(np.float32)
        val_norm = (data - self.min_val) / (self.max_val - self.min_val)
        return val_norm.astype(np.float32)


class Table(object):
    def __init__(self, data, table_name):
        # load data
        self.name = table_name
        self.data = data
        self.data_size_mb = self.data.values.nbytes / 1024 / 1024
        self.row_num = self.data.shape[0]
        self.col_num = len(self.data.columns)

        # parse columns
        self.columns = None
        self.parse_columns()
        L.info(f"build finished: {self}")

    def parse_columns(self):
        self.columns = OrderedDict([(col, Column(col, self.data[col])) for col in self.data.columns])

    def __repr__(self):
        return f"Table {self.name} : {self.row_num} rows, {self.data_size_mb:.2f}MB.\ncolumns:{[c.name for c in self.columns.values()]}"

    def normalize(self, scale=1):
        data = copy.deepcopy(self.data)
        for cname, col in self.columns.items():
            data[cname] = col.normalize(data[cname].values) * scale
        return data


def load_table(table_name: str, data_root: str, overwrite: bool = False) -> Table:
    obj_path = TEMP_ROOT / "obj"
    if not obj_path.exists():
        obj_path.mkdir()
    table_path = obj_path / f"{table_name}.table.pkl"
    if not overwrite and table_path.is_file():
        L.info("table exists, load...")
        with open(table_path, 'rb') as f:
            table = pickle.load(f)
        L.info(f"load finished: {table}")
        return table
    else:
        origin_path = TEMP_ROOT / "table" / f"{table_name}.pkl"
        csv_to_pkl(data_root, table_name)
        data = pd.read_pickle(origin_path)
        table = Table(data, table_name)
        L.info("write table to disk...")
        with open(table_path, 'wb') as f:
            pickle.dump(table, f, protocol=PKL_PROTO)
        return table


def csv_to_pkl(data_root: str, table_name: str):
    table_path = data_root / f"{table_name}.csv"
    temp_table_path = TEMP_ROOT / "table"
    if not temp_table_path.exists():
        temp_table_path.mkdir()
    pkl_path = temp_table_path / f"{table_name}.pkl"
    if pkl_path.is_file():
        return
    df = pd.read_csv(table_path)
    df.to_pickle(pkl_path)
    L.info("csv to pkl in path " + str(pkl_path))
