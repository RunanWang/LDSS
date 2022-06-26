import copy
import time
import random
import pickle
import numpy as np
import pandas as pd
from math import ceil
from utils.log import Log
from GenDataset.single.dataset.data_feat import Data
from utils.dtype import is_categorical
from GenDataset.single.workload.workload import query_2_triple
from GenDataset.single.estimator.estimator import Estimator, OPS
from constants import SAMPLE_GROUP_PAR, SAMPLE_IRR_COL_RATE, SAMPLE_DIS_COL_RATE, PKL_PROTO, TEMP_ROOT, mkdir

log = Log(__name__).get_logger()


class SamplingGroup(Estimator):
    def __init__(self, table, join_col=[]):
        random.seed(a=1, version=2)
        super(SamplingGroup, self).__init__(table=table)
        self.name_to_sample_unit = {}
        self.table = table
        self.digital_table = self.table.digitalize()
        for name, para in SAMPLE_GROUP_PAR.items():
            self.name_to_sample_unit[name] = SampleUnit(table, name, para['ratio'], para['seed'], para['replace'])
        log.info("All set Sampling Group has been established!")
        col_names = self.select_irrelevant_col()
        num = 1
        for col_name in col_names:
            name = "ir" + str(num) + col_name[:2]
            num += 1
            self.name_to_sample_unit[name] = self.gen_irr_col_sample_unit(col_name)
        log.info("All Irrelevant Col Sampling Group has been established!")
        col_names = self.select_distribute_col()
        for col_name in col_names:
            name = "dt" + str(num) + col_name[:2]
            num += 1
            seed = random.randint(1, 999999999)
            self.name_to_sample_unit[name] = StratifiedSamplingUnit(table, self.digital_table, col_name, name, seed)
        log.info("All Distribute Col Sampling Group has been established!")
        for col_name in join_col:
            name = "hs" + str(num) + col_name[:2]
            num += 1
            self.name_to_sample_unit[name] = HashSampleUnit(table, col_name, name, 0.01)
        log.info("All Hash Col Sampling Group has been established!")

    def select_irrelevant_col(self):
        # Select some irrelevant columns to establish sample on their selection.
        sel_target_num = ceil(len(self.table.data.columns) * SAMPLE_IRR_COL_RATE)
        corr = self.digital_table.corr()
        col_names = corr.columns.values.tolist()
        col_name_to_corr = {'col_name': [], 'corr_value': []}
        for col_name in col_names:
            corr[col_name][col_name] = 0
            col_name_to_corr['col_name'].append(col_name)
            col_name_to_corr['corr_value'].append(corr[col_name].max())
        selected_col = pd.DataFrame.from_dict(col_name_to_corr).sort_values('corr_value')['col_name'][
            :sel_target_num].to_list()
        return selected_col

    def select_distribute_col(self):
        # Select some irrelevant columns to establish sample on their selection.
        sel_target_num = ceil(len(self.table.data.columns) * SAMPLE_DIS_COL_RATE)
        table_mean = pd.DataFrame(self.digital_table.mean())
        table_std = pd.DataFrame(self.digital_table.std())
        table_cv = table_std / table_mean
        table_cv.reset_index(inplace=True)
        selected_col = table_cv.sort_values(0)["index"][:sel_target_num].to_list()
        return selected_col

    def gen_irr_col_sample_unit(self, col_name):
        quantile1 = self.digital_table[col_name].quantile(0.1)
        small_table = copy.deepcopy(self.table)
        small_table.data = self.table.data[self.digital_table[col_name] <= quantile1]
        r = len(small_table.data) / len(self.table.data) * 0.1
        if r > 0.08:
            small_table.data = self.table.data[self.digital_table[col_name] > quantile1]
            r = len(small_table.data) / len(self.table.data) * 0.1
        name = "ir_" + col_name[:2] + col_name[-3:]
        sample_unit = IrrSampleUnit(small_table, name, 0.1, 1, True)
        sample_unit.ratio = r
        sample_unit.row_num = self.table.row_num
        sample_unit.table = self.table
        log.debug(r)
        return sample_unit

    def query(self, query):
        card_dict = {}
        dur_dict = {}
        for name, unit in self.name_to_sample_unit.items():
            card_dict[name], dur_dict[name] = unit.query(query)
        return card_dict, dur_dict

    def like(self, like_str, str_col):
        card_dict = {}
        for name, unit in self.name_to_sample_unit.items():
            card_dict[name] = unit.like(like_str, str_col)
        return card_dict

    def feature(self, total_data_feat):
        feat_dict = {}
        for name, unit in self.name_to_sample_unit.items():
            feat_dict[name] = unit.feature(total_data_feat)
        return feat_dict

    def append_data(self, append_df):
        self.append_df = append_df
        for name, sample_unit in self.name_to_sample_unit.items():
            sample_unit.append_data(append_df)


class StratifiedSamplingUnit(object):
    def __init__(self, table, dtable, column_name, name, seed):
        self.id = name
        self.ratio = 0.02
        self.seed = seed
        self.replace = True
        self.column_name = column_name
        self.sample_units = []
        if is_categorical(table.columns[column_name].dtype) and table.columns[column_name].vocab_size <= 20:
            for i in range(0, table.columns[column_name].vocab_size):
                small_table = copy.deepcopy(table)
                small_table.data = table.data[dtable[column_name] == i]
                sample_unit = SampleUnit(small_table, column_name + str(i), self.ratio, seed, self.replace)
                sample_unit_rate = small_table.data.shape[0] / table.data.shape[0]
                sample_unit_info = {"unit": sample_unit, "rate": sample_unit_rate}
                self.sample_units.append(sample_unit_info)
        else:
            bin_quantile_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
            table_sort = table.data.sort_values(column_name)
            min_id = 0
            for bin_quantile in bin_quantile_range:
                max_id = int(bin_quantile * table.data.shape[0])
                small_table = copy.deepcopy(table)
                small_table.data = table_sort[min_id:max_id]
                sample_unit = SampleUnit(small_table, column_name + str(min_id), self.ratio, seed, self.replace)
                sample_unit_rate = small_table.data.shape[0] / table.data.shape[0]
                sample_unit_info = {"unit": sample_unit, "rate": sample_unit_rate}
                self.sample_units.append(sample_unit_info)
                min_id = max_id
            small_table = copy.deepcopy(table)
            small_table.data = table_sort[min_id:]
            sample_unit = SampleUnit(small_table, column_name + str(min_id), self.ratio, seed, self.replace)
            sample_unit_rate = small_table.data.shape[0] / table.data.shape[0]
            sample_unit_info = {"unit": sample_unit, "rate": sample_unit_rate}
            self.sample_units.append(sample_unit_info)
        self.row_num = table.row_num
        self.table = table

    def query(self, query):
        total_card = 0
        total_time = 0
        for sample_unit in self.sample_units:
            s_card, s_time = sample_unit["unit"].query(query)
            total_time += s_time
            total_card += sample_unit["rate"] * s_card
        return total_card, total_time

    def like(self, like_str, str_col):
        total_card = 0
        for sample_unit in self.sample_units:
            s_card = sample_unit["unit"].like(like_str, str_col)
            total_card += sample_unit["rate"] * s_card
        return total_card

    def feature(self, total_data_feat):
        sample_df_l = []
        for sample_unit in self.sample_units:
            sample_df_l.append(sample_unit["unit"].get_sample_df())
        sample_total_df = pd.concat(sample_df_l)
        sample_feature = {'ratio': self.ratio, 'replace': self.replace, 'seed': self.seed, "sample_type": 1}
        data = Data(sample_total_df, total_data_feat)
        data_feat = data.feature()
        for k, v in data_feat.items():
            sample_feature['sample_' + k] = v
        return sample_feature


class SampleUnit(object):
    def __init__(self, table, name, ratio, seed, replace):
        self.id = name
        self.sample = table.data.sample(frac=ratio, random_state=seed, replace=replace)
        self.sample_num = len(self.sample)
        self.row_num = table.row_num
        self.ratio = ratio
        self.replace = replace
        self.seed = seed
        self.table = table

    def query(self, query):
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        start_stmp = time.time()
        bitmap = np.ones(self.sample_num, dtype=bool)
        for c, o, v in zip(columns, operators, values):
            bitmap &= OPS[o](self.sample[c], v)
        if self.sample_num == 0:
            card = 0
        else:
            card = np.round((self.row_num / self.sample_num) * bitmap.sum())
        dur_ms = (time.time() - start_stmp) * 1e3
        return card, dur_ms

    def like(self, like_str, str_col):
        qualified = 0
        for index, row in self.sample.iterrows():
            if like_str in str(row[str_col]):
                qualified += 1
        card = np.round((self.row_num / self.sample_num) * qualified)
        return card

    def feature(self, total_data_feat):
        sample_feature = {'ratio': self.ratio, 'replace': self.replace, 'seed': self.seed, "sample_type": 0}
        data = Data(self.sample, total_data_feat)
        data_feat = data.feature()
        for k, v in data_feat.items():
            sample_feature['sample_' + k] = v
        return sample_feature

    def get_sample_df(self):
        return self.sample

    def append_data(self, append_df):
        self.append_df = append_df
        append_sample = append_df.sample(frac=self.ratio, random_state=self.seed, replace=self.replace)
        self.sample = pd.concat([self.sample, append_sample], ignore_index=True)
        self.sample_num = len(self.sample)
        self.row_num = self.row_num + len(append_df)


class IrrSampleUnit(object):
    def __init__(self, table, name, ratio, seed, replace):
        self.id = name
        self.sample = table.data.sample(frac=ratio, random_state=seed, replace=replace)
        self.sample_num = len(self.sample)
        self.row_num = table.row_num
        self.ratio = ratio
        self.replace = replace
        self.seed = seed
        self.table = table

    def query(self, query):
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        start_stmp = time.time()
        bitmap = np.ones(self.sample_num, dtype=bool)
        for c, o, v in zip(columns, operators, values):
            bitmap &= OPS[o](self.sample[c], v)
        if self.sample_num == 0:
            card = 0
        else:
            card = np.round((self.row_num / self.sample_num) * bitmap.sum())
        dur_ms = (time.time() - start_stmp) * 1e3
        return card, dur_ms

    def like(self, like_str, str_col):
        qualified = 0
        for index, row in self.sample.iterrows():
            if like_str in str(row[str_col]):
                qualified += 1
        card = np.round((self.row_num / self.sample_num) * qualified)
        return card

    def feature(self, total_data_feat):
        sample_feature = {'ratio': self.ratio, 'replace': self.replace, 'seed': self.seed, "sample_type": 3}
        data = Data(self.sample, total_data_feat)
        data_feat = data.feature()
        for k, v in data_feat.items():
            sample_feature['sample_' + k] = v
        return sample_feature

    def get_sample_df(self):
        return self.sample


class HashSampleUnit(object):
    def __init__(self, table, col, name, ratio):
        self.id = name
        col_df = table.data[col]
        def f(x): return hash(str(x))
        col_hash_df = col_df.apply(f)
        self.seed = col_hash_df.sample(n=1).to_list()[0] % (1 / ratio)
        self.sample = table.data[col_hash_df % (1 / ratio) == self.seed]
        self.sample_num = len(self.sample)
        self.row_num = table.row_num
        self.ratio = self.sample_num / self.row_num
        self.replace = True
        self.table = table
        self.seed = int(self.seed)

    def query(self, query):
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        start_stmp = time.time()
        bitmap = np.ones(self.sample_num, dtype=bool)
        for c, o, v in zip(columns, operators, values):
            bitmap &= OPS[o](self.sample[c], v)
        card = np.round((self.row_num / self.sample_num) * bitmap.sum())
        dur_ms = (time.time() - start_stmp) * 1e3
        return card, dur_ms

    def like(self, like_str, str_col):
        qualified = 0
        for index, row in self.sample.iterrows():
            if like_str in str(row[str_col]):
                qualified += 1
        card = np.round((self.row_num / self.sample_num) * qualified)
        return card

    def feature(self, total_data_feat):
        sample_feature = {'ratio': self.ratio, 'replace': self.replace, 'seed': self.seed, "sample_type": 2}
        data = Data(self.sample, total_data_feat)
        data_feat = data.feature()
        for k, v in data_feat.items():
            sample_feature['sample_' + k] = v
        return sample_feature

    def get_sample_df(self):
        return self.sample


def load_sample_group(table, dataset_name):
    sample_group_path = TEMP_ROOT / "sample_group"
    mkdir(sample_group_path)
    sample_group_path = sample_group_path / f"{dataset_name}-sg.pkl"
    if sample_group_path.is_file():
        log.info("sample group exists, load...")
        with open(sample_group_path, 'rb') as f:
            sample_group = pickle.load(f)
        log.info("load finished.")
        return sample_group
    else:
        sample_group = SamplingGroup(table)
        with open(sample_group_path, 'wb') as f:
            pickle.dump(sample_group, f, protocol=PKL_PROTO)
        return sample_group


def load_sample_feature(sg: SamplingGroup, dataset_name: str, total_data):
    sample_group_path = TEMP_ROOT / "sample_group"
    mkdir(sample_group_path)
    sample_feature_path = sample_group_path / f"{dataset_name}-sf.pkl"
    if sample_feature_path.is_file():
        log.info("sample features exists, load...")
        with open(sample_feature_path, 'rb') as f:
            sample_feat = pickle.load(f)
        log.info("load finished.")
        return sample_feat
    else:
        sample_feat = sg.feature(total_data)
        with open(sample_feature_path, 'wb') as f:
            pickle.dump(sample_feat, f, protocol=PKL_PROTO)
        return sample_feat
