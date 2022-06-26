import copy
import time
import random
import pickle
import numpy as np
import pandas as pd
from utils.log import Log
from GenDataset.multi.dataset.data import Data
from GenDataset.multi.dataset.table import Database
from GenDataset.multi.workload.workload import query_2_triple
from GenDataset.multi.estimator.estimator import Estimator, OPS
from constants import MULTI_SAMPLE_GROUP_PAR, SAMPLE_DIS_NUM, SAMPLE_HASH_RATE, TEMP_ROOT, mkdir, PKL_PROTO, SAMPLE_HASH_SEED

log = Log(__name__).get_logger()


class SamplingGroup(Estimator):
    def __init__(self, database):
        random.seed(a=1, version=2)
        super(SamplingGroup, self).__init__(database=database)
        self.name_to_sample_unit = {}
        self.database = database
        self.digital_database = {}
        for table_name, table in self.database.table.items():
            self.digital_database[table_name] = self.database.table[table_name].data
        for table_name, table in self.database.table.items():
            self.name_to_sample_unit[table_name] = {}
            for name, para in MULTI_SAMPLE_GROUP_PAR.items():
                u = SampleUnit(table, name, para['ratio'], para['seed'],
                               para['replace'], self.database.join_col, table_name)
                self.name_to_sample_unit[table_name][name] = u
        log.info("All set Sampling Group has been established!")
        for table_name, table in self.database.table.items():
            col_names = self.select_distribute_col(table_name)
            for col_name in col_names:
                name = f"dt_{table_name}_{col_name}"
                seed = random.randint(1, 999999999)
                u = StratifiedSamplingUnit(table, col_name, name, seed, self.database.join_col, table_name)
                self.name_to_sample_unit[table_name][name] = u
        ratio = SAMPLE_HASH_RATE
        log.info("All Distribute Col Sampling Group has been established!")
        seed = self.database.table[self.database.base_table].data.sample(
            n=1)[self.database.join_col[self.database.base_table]].tolist()[0]
        seed = hash(seed) % (1 / ratio)
        seed = SAMPLE_HASH_SEED
        for table_name, col_name in self.database.join_col.items():
            name = f"hs_{table_name}"
            u = HashSampleUnit(self.database.table[table_name], col_name, name,
                               ratio, seed, self.database.join_col, table_name)
            self.name_to_sample_unit[table_name][name] = u
        log.info("All Hash Col Sampling Group has been established!")

    def select_distribute_col(self, table_name: str):
        # Select some irrelevant columns to establish sample on their selection.
        sel_target_num = SAMPLE_DIS_NUM
        table_mean = pd.DataFrame(self.digital_database[table_name].mean())
        table_std = pd.DataFrame(self.digital_database[table_name].std())
        table_cv = table_std / table_mean
        table_cv.reset_index(inplace=True)
        selected_col = table_cv.sort_values(0)["index"][:sel_target_num].to_list()
        return selected_col

    def query(self, query):
        card_dict = {}
        dur_dict = {}
        total_card = {None: None}
        result_dict = {}
        cost_dict = {}
        for table_name, join_column in query.join:
            card_dict[table_name] = {}
            dur_dict[table_name] = {}
            for name, unit in self.name_to_sample_unit[table_name].items():
                card_dict[table_name][name], dur_dict[table_name][name] = unit.query(query, table_name)
        # no hash
        for table_name, join_column in query.join:
            temp_total_card = {}
            for name, unit in self.name_to_sample_unit[table_name].items():
                if "hs" in name:
                    continue
                for part_name, part_card in total_card.items():
                    if part_name is None:
                        temp_total_card[f"{table_name}_{name}"] = card_dict[table_name][name]
                    else:
                        join_df = card_dict[table_name][name].join(part_card, lsuffix='_left', rsuffix='_right')
                        join_df = join_df.fillna(0)
                        result_temp = pd.DataFrame(join_df["size_left"] * join_df['size_right'], columns=["size"])
                        result_temp = result_temp.drop(result_temp[result_temp["size"] == 0].index)
                        temp_total_card[f"{part_name}_{table_name}_{name}"] = result_temp
            total_card = temp_total_card
        for part_name, part_card in total_card.items():
            result_dict[part_name] = part_card.sum()["size"]
        # hash
        total_card = {None: None}
        for table_name, join_column in query.join:
            temp_total_card = {}
            for name, unit in self.name_to_sample_unit[table_name].items():
                if "hs" not in name:
                    continue
                for part_name, part_card in total_card.items():
                    if part_name is None:
                        temp_total_card[f"{table_name}_{name}"] = card_dict[table_name][name]
                    else:
                        join_df = card_dict[table_name][name].join(part_card, lsuffix='_left', rsuffix='_right')
                        join_df = join_df.fillna(0)
                        result_temp = pd.DataFrame(join_df["size_left"] *
                                                   join_df['size_right'] * SAMPLE_HASH_RATE, columns=["size"])
                        result_temp = result_temp.drop(result_temp[result_temp["size"] == 0].index)
                        temp_total_card[f"{part_name}_{table_name}_{name}"] = result_temp
            total_card = temp_total_card
        for part_name, part_card in total_card.items():
            result_dict[part_name] = part_card.sum()["size"]
        return result_dict, cost_dict

    def feature(self):
        feat_dict = {}
        for table_name, table in self.database.table.items():
            total_data = Data(table.data, tname=table_name)
            feat_dict[table_name] = {}
            for name, unit in self.name_to_sample_unit[table_name].items():
                feat_dict[table_name][name] = unit.feature(total_data)
        return feat_dict


class StratifiedSamplingUnit(object):
    def __init__(self, table, column_name, name, seed, join_col, table_name=None):
        self.id = name
        self.table_name = table_name
        self.ratio = 0.02
        self.seed = seed
        self.replace = True
        self.sample_units = []
        self.join_col = join_col

        bin_quantile_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        table_sort = table.data.sort_values(column_name)
        min_id = 0
        for bin_quantile in bin_quantile_range:
            max_id = int(bin_quantile * table.data.shape[0])
            small_table = copy.deepcopy(table)
            small_table.data = table_sort[min_id:max_id]
            sample_unit = SampleUnit(small_table, column_name + str(min_id),
                                     self.ratio, seed, self.replace, self.join_col)
            sample_unit_rate = small_table.data.shape[0] / table.data.shape[0]
            sample_unit_info = {"unit": sample_unit, "rate": sample_unit_rate}
            self.sample_units.append(sample_unit_info)
            min_id = max_id
        small_table = copy.deepcopy(table)
        small_table.data = table_sort[min_id:]
        sample_unit = SampleUnit(small_table, column_name + str(min_id), self.ratio, seed, self.replace, self.join_col)
        sample_unit_rate = small_table.data.shape[0] / table.data.shape[0]
        sample_unit_info = {"unit": sample_unit, "rate": sample_unit_rate}
        self.sample_units.append(sample_unit_info)

        self.row_num = table.row_num
        self.table = table

    def query(self, query, table_name):
        total_card = pd.DataFrame(columns=["size"])
        total_time = 0
        for sample_unit in self.sample_units:
            s_card, s_time = sample_unit["unit"].query(query, table_name)
            total_time += s_time
            temp_df = total_card.join(sample_unit["rate"] * s_card, how="outer", lsuffix='_left', rsuffix='_right')
            temp_df = temp_df.fillna(0)
            total_card = pd.DataFrame(temp_df["size_left"] + temp_df['size_right'], columns=["size"])
        return total_card, total_time

    def feature(self, total_data_feat):
        sample_df_l = []
        for sample_unit in self.sample_units:
            sample_df_l.append(sample_unit["unit"].get_sample_df())
        sample_total_df = pd.concat(sample_df_l)
        sample_feature = {f'of_{self.table_name}_ratio': self.ratio, f'of_{self.table_name}_replace': self.replace,
                          f'id_{self.table_name}_seed': self.seed, f"of_{self.table_name}_sample_type": 1}
        data = Data(sample_total_df, total_data_feat, self.table_name)
        data_feat = data.feature()
        for k, v in data_feat.items():
            sample_feature[k] = v
        return sample_feature


class SampleUnit(object):
    def __init__(self, table, name, ratio, seed, replace, join_col, table_name=None):
        self.id = name
        self.table_name = table_name
        self.sample = table.data.sample(frac=ratio, random_state=seed, replace=replace)
        self.sample_num = len(self.sample)
        self.row_num = table.row_num
        self.ratio = ratio
        self.replace = replace
        self.seed = seed
        self.table = table
        self.join_col = join_col

    def query(self, query, table_name):
        columns, operators, values = query_2_triple(query, table_name, with_none=False, split_range=False)
        start = time.time()
        bitmap = np.ones(self.sample_num, dtype=bool)
        for c, o, v in zip(columns, operators, values):
            bitmap &= OPS[o](self.sample[c], v)
        if self.sample_num == 0:
            card = pd.DataFrame(columns=["size"])
        else:
            card = pd.DataFrame(self.sample[bitmap].groupby(by=self.join_col[table_name]).size(), columns=["size"])
            card["size"] = card["size"] / self.sample_num
        dur_ms = (time.time() - start) * 1e3
        return card, dur_ms

    def feature(self, total_data_feat):
        sample_feature = {f'of_{self.table_name}_ratio': self.ratio, f'of_{self.table_name}_replace': self.replace,
                          f'id_{self.table_name}_seed': self.seed, f"of_{self.table_name}_sample_type": 3}
        data = Data(self.sample, total_data_feat, self.table_name)
        data_feat = data.feature()
        for k, v in data_feat.items():
            sample_feature[k] = v
        return sample_feature

    def get_sample_df(self):
        return self.sample


class HashSampleUnit(object):
    def __init__(self, table, col, name, ratio, seed, join_col, table_name=None):
        self.id = name
        self.table_name = table_name
        col_df = table.data[col]
        def f(x): return hash(x)
        col_hash_df = col_df.apply(f)
        self.seed = seed
        self.sample = table.data[col_hash_df % (1 / ratio) == self.seed]
        self.sample_num = len(self.sample)
        self.row_num = table.row_num
        self.ratio = self.sample_num / self.row_num
        self.replace = True
        self.table = table
        self.seed = int(self.seed)
        self.join_col = join_col

    def query(self, query, table_name):
        columns, operators, values = query_2_triple(query, table_name, with_none=False, split_range=False)
        start = time.time()
        bitmap = np.ones(self.sample_num, dtype=bool)
        for c, o, v in zip(columns, operators, values):
            bitmap &= OPS[o](self.sample[c], v)
        if self.sample_num == 0:
            card = pd.DataFrame(columns=["size"])
        else:
            card = pd.DataFrame(self.sample[bitmap].groupby(by=self.join_col[table_name]).size(), columns=["size"])
            card["size"] = card["size"] / self.sample_num
        dur_ms = (time.time() - start) * 1e3
        return card, dur_ms

    def feature(self, total_data_feat):
        sample_feature = {f'of_{self.table_name}_ratio': self.ratio, f'of_{self.table_name}_replace': self.replace,
                          f'id_{self.table_name}_seed': self.seed, f"of_{self.table_name}_sample_type": 2}
        data = Data(self.sample, total_data_feat, self.table_name)
        data_feat = data.feature()
        for k, v in data_feat.items():
            sample_feature[k] = v
        return sample_feature

    def get_sample_df(self):
        return self.sample


def load_sample_group(database: Database, dataset_name: str):
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
        sample_group = SamplingGroup(database)
        with open(sample_group_path, 'wb') as f:
            pickle.dump(sample_group, f, protocol=PKL_PROTO)
        return sample_group


def load_sample_feature(sg: SamplingGroup, dataset_name: str):
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
        sample_feat = sg.feature()
        with open(sample_feature_path, 'wb') as f:
            pickle.dump(sample_feat, f, protocol=PKL_PROTO)
        return sample_feat
