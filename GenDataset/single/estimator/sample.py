import time
from utils.log import Log
import numpy as np
import pandas as pd
from GenDataset.single.estimator.estimator import Estimator, OPS
from GenDataset.single.workload.workload import query_2_triple

L = Log(__name__).get_logger()


class Sampling(Estimator):
    def __init__(self, table, ratio, seed):
        super(Sampling, self).__init__(table=table, ratio=ratio, seed=seed)
        self.sample = table.data.sample(frac=ratio, random_state=seed)
        self.sample_num = len(self.sample)
        self.row_num = self.table.row_num
        self.ratio = ratio
        self.seed = seed

    def query(self, query):
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        start_stmp = time.time()
        bitmap = np.ones(self.sample_num, dtype=bool)
        for c, o, v in zip(columns, operators, values):
            bitmap &= OPS[o](self.sample[c], v)
        card = np.round((self.table.row_num / self.sample_num) * bitmap.sum())
        dur_ms = (time.time() - start_stmp) * 1e3
        return card, dur_ms

    def query2(self, query):
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        dur = 0
        bitmap = np.ones(self.sample_num, dtype=bool)
        df = self.sample
        for c, o, v in zip(columns, operators, values):
            dfc = df[c]
            start_stmp = time.time()
            bitmap = OPS[o](dfc, v)
            dur += time.time() - start_stmp
            df = df[bitmap]
            bitmap = bitmap[bitmap]
        if self.sample_num == 0:
            card = 0
        else:
            card = np.round((self.row_num / self.sample_num) * bitmap.sum())
        dur_ms = dur * 1e3
        return card, dur_ms

    def like(self, like_str, str_col):
        qualified = 0
        for index, row in self.sample.iterrows():
            if like_str in str(row[str_col]):
                qualified += 1
        card = np.round((self.table.row_num / self.sample_num) * qualified)
        return card

    def append_data(self, append_df):
        self.append_df = append_df
        append_sample = append_df.sample(frac=self.ratio, random_state=self.seed)
        self.sample = pd.concat([self.sample, append_sample], ignore_index=True)
        self.sample_num = len(self.sample)
        self.row_num = self.row_num + len(append_df)
        return self
