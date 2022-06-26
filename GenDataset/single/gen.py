import pandas as pd
from multiprocessing import Process, Manager, Lock

from utils.log import Log
from constants import DATASET_PATH
from GenDataset.single.workload.gen_label import generate_label_for_query
from GenDataset.single.dataset.dataset import load_table
from GenDataset.single.dataset.data_feat import load_data
from GenDataset.single.workload.workload import load_gen_queryset, query_2_vec, query_2_sql, query_2_histogram_vec
from GenDataset.single.estimator.sample_group import load_sample_feature, load_sample_group

L = Log(__name__).get_logger()


def cal_one_query(t, query, query_num, sample_group, sample_feature, q_f, r_list, q_list, record_lock):
    rec_for_one_query = []
    gt_card = generate_label_for_query(t, query)
    card, cost = sample_group.query(query)
    query_vec = query_2_vec(query, t)
    query_sql = query_2_sql(query, t, aggregate=False, dbms='postgres')
    query_vec.update(q_f)
    # query_vec['id_sql'] = query_sql
    query_vec['id_query_no'] = query_num
    for sample_name, sample_card in card.items():
        record = {}
        record.update(query_vec)
        record['id_sample_name'] = sample_name
        record.update(sample_feature[sample_name])
        record['label_GT_card'] = gt_card
        record['label_est_card'] = sample_card
        record['label_row_num'] = t.row_num
        record['label_cost_time'] = cost[sample_name]
        rec_for_one_query.append(record)
    record_lock.acquire()
    r_list.extend(rec_for_one_query)
    q_list.append({"query_no": query_num, "query": query_sql, "cardinality_true": gt_card})
    record_lock.release()


def gen_single_dataset(dataset_name, workload_name):
    # Load workload
    queryset = load_gen_queryset(dataset_name)

    # Load table
    table = load_table(dataset_name)

    # Load samples
    total_data = load_data(dataset_name, table)
    sampling_group = load_sample_group(table, dataset_name)
    sample_features = load_sample_feature(sampling_group, dataset_name, total_data)
    # total_data = Data(table.data)
    # # table_feats = total_data.feature()
    # sampling_group = SamplingGroup(table)
    # sample_features = sampling_group.feature(total_data)

    for group, queries in queryset.items():
        L.info(group + " group is being processed!")
        p_list = []
        lock = Lock()
        manager = Manager()
        record_list = manager.list()
        query_list = manager.list()
        for i, q in enumerate(queries):
            q_feat = query_2_histogram_vec(q, total_data)
            p = Process(target=cal_one_query,
                        args=(table, q, i, sampling_group, sample_features, q_feat, record_list, query_list, lock,))
            p.start()
            p_list.append(p)
            if i % 100 == 0:
                for p in p_list:
                    p.join()
            if i % 1000 == 0:
                L.info("Query of group " + group + " - {:>6d} / {}".format(i, len(queries)))
        L.info("Saving records of group " + group)
        pkl_path = DATASET_PATH / f"{dataset_name}-{workload_name}-{group}.pkl"
        df = pd.DataFrame(list(record_list))
        record_list[:] = []
        df.to_pickle(pkl_path)
        csv_path = DATASET_PATH / f"q-{dataset_name}-{workload_name}.csv"
        df = pd.DataFrame(list(query_list))
        query_list[:] = []
        df.to_csv(csv_path)
