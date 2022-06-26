import copy
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd

from constants import FEAT_MAX_BINS, FEAT_MAX_CBINS, TEMP_ROOT, PKL_PROTO
from utils.dtype import is_categorical
from utils.log import Log
from GenDataset.single.dataset.dataset import Table

L = Log(__name__).get_logger()


def list_to_str(l: list) -> str:
    ret = ""
    for s in l:
        ret += str(s)
    return ret


class Col(object):
    def __init__(self, name, data, bins=[]):
        self.name = name
        self.data = data
        self.type = data.dtype

        # parse vocabulary
        self.vocab, self.has_nan = self.parse_vocab()
        self.vocab_size = len(self.vocab)
        self.min_val = self.vocab[1] if self.has_nan else self.vocab[0]
        self.max_val = self.vocab[-1]
        self.row_num = self.data.shape[0]

        # get normalize column (dataframe)
        self.norm_data = pd.DataFrame(self.normalize(self.data))

        # bins
        if len(bins) == 0:
            self.cate_to_sel = self.gen_bins()
        else:
            self.bins = bins
            self.cate_to_sel = self.cal_bins()

    def parse_vocab(self):
        is_nan = pd.isnull(self.data)
        contains_nan = np.any(is_nan)
        vs = np.sort(np.unique(self.data[~is_nan]))
        if contains_nan:
            vs = np.insert(vs, 0, np.nan)
        return vs, contains_nan

    def digitalize(self, data):
        """Transforms data values into integers using a Column's vocabulary"""
        # pd.Categorical() does not allow categories be passed in an array
        # containing np.nan.  It makes it a special case to return code -1
        # for NaN values.
        if self.has_nan:
            bin_ids = pd.Categorical(data, categories=self.vocab[1:]).codes
            # Since nan/nat bin_id is supposed to be 0 but pandas returns -1, just
            # add 1 to everybody
            bin_ids = bin_ids + 1
        else:
            # This column has no nan or nat values.
            bin_ids = pd.Categorical(data, categories=self.vocab).codes

        bin_ids = bin_ids.astype(np.int32, copy=False)
        assert (bin_ids >= 0).all(), (self, data, bin_ids)
        return bin_ids

    def normalize(self, data):
        """Normalize data to range [0, 1]"""
        min_val = self.min_val
        max_val = self.max_val
        # if column is not numerical, use descretized value
        if is_categorical(self.type):
            data = self.digitalize(data)
            min_val = 0
            max_val = self.vocab_size - 1
        data = np.array(data, dtype=np.float32)
        if min_val >= max_val:
            L.warning(f"column {self.name} has min value {min_val} >= max value{max_val}")
            return np.zeros(len(data)).astype(np.float32)
        val_norm = (data - min_val) / (max_val - min_val)
        return val_norm.astype(np.float32)

    def gen_bins(self) -> dict:
        """
            Generate queries on this column by dividing possible values into range.
            :return: cardinality on each generated query.
        """
        cate_to_sel = {}  # cate_name to cate_sel
        data = copy.deepcopy(self.data)
        data = pd.DataFrame(data)
        if is_categorical(self.type):
            cates = []
            # group into bins
            df_name_count = data.groupby([self.name], as_index=False).size()
            df_name_count.sort_values(by=["size"])
            np_name_count = df_name_count.values
            for pair in np_name_count:  # pair: name card
                cates.append([pair[0]])
                cate_to_sel[pair[0]] = pair[1] / self.row_num
            # merge bins if too large
            if len(cates) > FEAT_MAX_CBINS:
                to_merge = len(cates) - FEAT_MAX_CBINS
                if FEAT_MAX_CBINS == -1:
                    to_merge = 0
                while to_merge > 0:
                    # merge bins
                    cate_1 = cates.pop(0)
                    cate_2 = cates.pop(0)
                    cates.append(cate_1 + cate_2)
                    # merge name and selectivity
                    cate_name1 = list_to_str(cate_1)
                    cate_name2 = list_to_str(cate_2)
                    cate_to_sel[cate_name1 + cate_name2] = cate_to_sel[cate_name1] + cate_to_sel[cate_name2]
                    cate_to_sel.pop(cate_name1)
                    cate_to_sel.pop(cate_name2)
                    to_merge -= 1
            self.bins = cates
        else:
            # for continues: separate by range.
            self.bins = []
            range_length = (self.max_val - self.min_val) / FEAT_MAX_BINS
            for i in range(1, FEAT_MAX_BINS):
                x_ranges = [self.min_val + range_length * (i - 1), self.min_val + range_length * i]
                self.bins.append(x_ranges)
                bin_name = list_to_str(x_ranges)
                bool_df = (data[self.name] >= (self.min_val + range_length * (i - 1))) & (
                    data[self.name] < (self.min_val + range_length * i))
                cate_to_sel[bin_name] = len(data[bool_df]) / self.row_num
            x_ranges = [self.min_val + range_length * (FEAT_MAX_BINS - 1), self.min_val + range_length * FEAT_MAX_BINS]
            self.bins.append(x_ranges)
            bin_name = list_to_str(x_ranges)
            bool_df = (data[self.name] >= (self.min_val + range_length * (FEAT_MAX_BINS - 1))) & (
                data[self.name] <= (self.min_val + range_length * FEAT_MAX_BINS))
            cate_to_sel[bin_name] = len(data[bool_df]) / self.row_num
        return cate_to_sel

    def cal_bins(self) -> dict:
        """
        calculate selectivity by informed bins
        :return:
        """
        bin_to_sel = {}  # cate_name to cate_sel
        cate_to_sel = {}  # cate_name to cate_sel
        data = copy.deepcopy(self.data)
        data = pd.DataFrame(data)
        if is_categorical(self.type):
            # group by all of data
            df_name_count = data.groupby([self.name], as_index=False).size()
            np_name_count = df_name_count.values
            for pair in np_name_count:  # pair: name card
                cate_to_sel[pair[0]] = pair[1] / self.row_num
            # foreach group, cal sel
            for l in self.bins:
                name = list_to_str(l)
                sel = 0
                for cate in l:
                    try:
                        sel += cate_to_sel[cate]
                    except KeyError:
                        sel += 0
                bin_to_sel[name] = sel
        else:
            # for continues: separate by range.
            x_ranges = []
            bin_name = ""
            for x_ranges in self.bins:
                bin_name = list_to_str(x_ranges)
                bool_df = (data[self.name] >= x_ranges[0]) & (data[self.name] < x_ranges[1])
                bin_to_sel[bin_name] = len(data[bool_df]) / self.row_num
            bool_df = (data[self.name] == x_ranges[1])
            bin_to_sel[bin_name] += len(data[bool_df]) / self.row_num
        return bin_to_sel

    def gen_feat(self) -> dict:
        """
            Generate feature of this column.
            Normalize data and gets its statistics info and percentile info as its feature.
            :return: features dict
        """
        sample_feature = {}
        mean_val = self.norm_data.mean()[0]
        std_val = self.norm_data.std()[0]
        sample_feature[self.name + '_mean'] = mean_val
        sample_feature[self.name + '_std'] = std_val
        for k, v in self.cate_to_sel.items():
            sample_feature[self.name + "_" + str(k)] = v
        return sample_feature


class Data(object):
    def __init__(self, data: pd.DataFrame, ref_data=None):
        L.info(f"Building feature. Data Length:{len(data)}")
        # load data
        self.data = data
        self.row_num = self.data.shape[0]
        self.col_num = len(self.data.columns)

        # parse columns
        if ref_data is None:
            self.has_ref = False
            self.columns = OrderedDict([(col, Col(col, self.data[col])) for col in self.data.columns])
        else:
            L.info(f"For test append table size4: {len(ref_data.data)}")
            self.has_ref = True
            self.columns = OrderedDict([(col, Col(col, self.data[col], ref_data.columns[col].bins))
                                       for col in self.data.columns])
            self.ref_feat = ref_data.feature()
        L.info("Build finished")

    def normalize(self, scale=1):
        data = copy.deepcopy(self.data)
        for cname, col in self.columns.items():
            data[cname] = col.normalize(data[cname].values) * scale
        return data

    def digitalize(self):
        data = copy.deepcopy(self.data)
        for cname, col in self.columns.items():
            if is_categorical(col.type):
                data[cname] = col.digitalize(data[cname].values)
            elif col.has_nan:
                data[cname].fillna(0, inplace=True)
        return data

    def reset_voca(self, ref_table):
        for cname, col in self.columns.items():
            col.minval = ref_table.columns[cname].minval
            col.maxval = ref_table.columns[cname].maxval
            col.vocab = ref_table.columns[cname].vocab
            col.vocab_size = ref_table.columns[cname].vocab_size

    def feature(self):
        table_feature = {}
        col_name_done = []
        corr = self.digitalize().corr()
        col_names = corr.columns.values.tolist()
        for col_name1 in col_names:
            for col_name2 in col_names:
                if col_name1 + col_name2 in col_name_done:
                    pass
                elif col_name2 + col_name1 in col_name_done:
                    pass
                elif col_name1 == col_name2:
                    pass
                else:
                    table_feature["co_" + col_name1 + "_" + col_name2] = corr[col_name1][col_name2]
                    col_name_done.append(col_name1 + col_name2)
        if self.has_ref:
            for cname, col in self.columns.items():
                col_feat = col.gen_feat()
                for k, v in col_feat.items():
                    table_feature[k] = self.ref_feat[k] - v
        else:
            for cname, col in self.columns.items():
                col_feat = col.gen_feat()
                table_feature.update(col_feat)
        L.info("Length of table feature: " + str(len(table_feature)))
        return table_feature


def load_data(table_name: str, table: Table, overwrite: bool = False) -> Data:
    obj_path = TEMP_ROOT / "obj"
    if not obj_path.exists():
        obj_path.mkdir()
    table_path = obj_path / f"{table_name}.data.pkl"
    if not overwrite and table_path.is_file():
        L.info("data exists, load...")
        with open(table_path, 'rb') as f:
            data = pickle.load(f)
        L.info(f"load data finished: {table_name}")
        return data
    else:
        data = Data(table.data)
        L.info("write data to disk...")
        with open(table_path, 'wb') as f:
            pickle.dump(data, f, protocol=PKL_PROTO)
        return data
