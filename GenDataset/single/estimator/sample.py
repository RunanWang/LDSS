import time
from utils.log import Log
import numpy as np
from GenDataset.single.estimator.estimator import Estimator, OPS
from GenDataset.single.workload.workload import query_2_triple

L = Log().Log(__name__).get_logger()


class Sampling(Estimator):
    def __init__(self, table, ratio, seed):
        super(Sampling, self).__init__(table=table, ratio=ratio, seed=seed)
        self.sample = table.data.sample(frac=ratio, random_state=seed)
        self.sample_num = len(self.sample)

    def query(self, query):
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        start_stmp = time.time()
        bitmap = np.ones(self.sample_num, dtype=bool)
        for c, o, v in zip(columns, operators, values):
            bitmap &= OPS[o](self.sample[c], v)
        card = np.round((self.table.row_num / self.sample_num) * bitmap.sum())
        dur_ms = (time.time() - start_stmp) * 1e3
        return card, dur_ms

    def like(self, like_str, str_col):
        qualified = 0
        for index, row in self.sample.iterrows():
            if like_str in str(row[str_col]):
                qualified += 1
        card = np.round((self.table.row_num / self.sample_num) * qualified)
        return card
