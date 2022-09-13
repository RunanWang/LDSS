import copy
import pickle

import numpy as np
import pandas as pd

from constants import FEAT_MAX_BINS, FEAT_MAX_CBINS, TEMP_ROOT, PKL_PROTO
from utils.dtype import is_categorical
from utils.log import Log

L = Log(__name__).get_logger()


def list_to_str(l: list) -> str:
    ret = ""
    for s in l:
        ret += str(s)
    return ret


class FeatColumn(object):
    def __init__(self, col_name: str, df_col: pd.DataFrame, ref_col=None):
        self.name = col_name
        self.df_col = df_col
        self.type = self.df_col.dtypes
        self.row_num = self.df_col.shape[0]

        # parse vocab
        if ref_col is None:
            self.vocab, self.has_nan = self.parse_vocab()
        else:
            self.vocab = ref_col.vocab
            self.has_nan = ref_col.has_nan
        self.vocab_size = len(self.vocab)

        # numerical & normalization
        self.numerical()
        self.min_val, self.max_val = self.get_min_max()
        self.df_norm_col = pd.DataFrame(self.normal())

        # histogram
        if ref_col is None:
            self.bins = []
            self.cate_to_sel = self.gen_bins()
        else:
            self.bins = ref_col.bins
            self.cate_to_sel = self.cal_bins()

    def parse_vocab(self):
        is_nan = pd.isnull(self.df_col)
        contains_nan = np.any(is_nan)
        vs = np.sort(np.unique(self.df_col[~is_nan]))
        if contains_nan:
            vs = np.insert(vs, 0, np.nan)
        return vs, contains_nan

    def numerical(self):
        """Transforms data values into integers using a Column's vocabulary"""
        if is_categorical(self.type):
            if self.has_nan:
                bin_ids = pd.Categorical(self.df_col, categories=self.vocab[1:]).codes
                bin_ids = bin_ids + 1
            else:
                bin_ids = pd.Categorical(self.df_col, categories=self.vocab).codes
            bin_ids = bin_ids.astype(np.int32, copy=False)
            self.df_col = pd.Series(bin_ids)
        elif self.has_nan:
            min_val = self.vocab[1] if self.has_nan else self.vocab[0]
            self.df_col.fillna(min_val - 1)
        self.df_col = self.df_col.reset_index(drop=True)

    def get_min_max(self):
        if is_categorical(self.type):
            min_val = 0
            max_val = self.vocab_size - 1
        else:
            min_val = self.vocab[1] if self.has_nan else self.vocab[0]
            max_val = self.vocab[-1]
        return min_val, max_val

    def normal(self):
        """Normalize data to range [0, 1]"""
        copy_df_col = copy.deepcopy(self.df_col)
        copy_df_col = np.array(copy_df_col, dtype=np.float32)
        if self.min_val >= self.max_val:
            return np.zeros(len(copy_df_col)).astype(np.float32)
        val_norm = (copy_df_col - self.min_val) / (self.max_val - self.min_val)
        return val_norm.astype(np.float32)

    def gen_bins(self) -> dict:
        """
            Generate queries on this column by dividing possible values into range.
            :return: cardinality on each generated query.
        """
        cate_to_sel = {}  # cate_name to cate_sel
        data = copy.deepcopy(self.df_col)
        data = pd.DataFrame(data, columns=[self.name])
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
        data = copy.deepcopy(self.df_col)
        data = pd.DataFrame(data, columns=[self.name])
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
        mean_val = self.df_norm_col.mean()[0]
        std_val = self.df_norm_col.std()[0]
        sample_feature[self.name + '_mean'] = mean_val
        sample_feature[self.name + '_std'] = std_val
        for k, v in self.cate_to_sel.items():
            sample_feature[self.name + "_" + str(k)] = v
        return sample_feature


def get_corr(columns_dict):
    table_feature = {}
    col_name_done = []
    numerical_data = pd.DataFrame()
    for col, featColumn in columns_dict.items():
        numerical_data[col] = featColumn.df_col
    corr = numerical_data.corr()
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
    return table_feature


class Feature(object):
    def __init__(self, df_data: pd.DataFrame, df_relative: pd.DataFrame, df_total: pd.DataFrame):
        self.total_columns_dict = {}
        self.rel_columns_dict = {}
        self.data_columns_dict = {}
        for col in df_total.columns:
            self.total_columns_dict[col] = FeatColumn(col, df_total[col])
            self.data_columns_dict[col] = FeatColumn(col, df_data[col], self.total_columns_dict[col])
            self.rel_columns_dict[col] = FeatColumn(col, df_relative[col], self.total_columns_dict[col])

    def feature(self):
        feat_dict = {}
        feat_data_dict = {}
        feat_rel_dict = {}

        for col, _ in self.total_columns_dict.items():
            feat_data_dict.update(self.data_columns_dict[col].gen_feat())
            feat_data_dict.update(get_corr(self.data_columns_dict))
            feat_rel_dict.update(self.rel_columns_dict[col].gen_feat())
            feat_rel_dict.update(get_corr(self.rel_columns_dict))

        for k, v in feat_rel_dict.items():
            feat_dict[k] = v - feat_data_dict[k]
        L.info(f"Generated feature size: {len(feat_dict)}")
        return feat_dict


class Histogram(object):
    def __init__(self, df_total: pd.DataFrame):
        self.col_2_histo_bin = {}
        self.col_2_vocab = {}
        self.col_2_type = {}
        for col in df_total.columns:
            feat_column = FeatColumn(col, df_total[col])
            self.col_2_histo_bin[col] = feat_column.bins
            str_2_id = {}
            if is_categorical(feat_column.type):
                for i in range(len(feat_column.vocab)):
                    str_2_id[feat_column.vocab[i]] = i
                self.col_2_vocab[col] = str_2_id
            self.col_2_type[col] = feat_column.type


def load_histo(table_name: str, df_total: pd.DataFrame, overwrite: bool = False) -> Histogram:
    obj_path = TEMP_ROOT / "obj"
    if not obj_path.exists():
        obj_path.mkdir()
    table_path = obj_path / f"{table_name}.histo.pkl"
    if not overwrite and table_path.is_file():
        L.info("Histo exists, load...")
        with open(table_path, 'rb') as f:
            data = pickle.load(f)
        L.info(f"load finished: {table_name}")
        return data
    else:
        L.info("Histo exists, load...")
        data = Histogram(df_total)
        L.info("write data to disk...")
        with open(table_path, 'wb') as f:
            pickle.dump(data, f, protocol=PKL_PROTO)
        return data
