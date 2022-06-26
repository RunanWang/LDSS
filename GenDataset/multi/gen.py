from multiprocessing import Process, Manager, Lock
import pandas as pd

from constants import DATASET_PATH, TEMP_DATASET_PATH, DATA_ROOT, WORKLOAD_ROOT
from GenDataset.multi.dataset.table import Database
from GenDataset.multi.workload.gen_workload import generate_workload_from_sql
from GenDataset.multi.workload.workload import query_2_sql, query_2_vec, gen_row_num, query_2_histogram_vec
from GenDataset.multi.estimator.sample import load_sample_group, load_sample_feature
from GenDataset.multi.estimator.estimator import Oracle
from utils.log import Log

L = Log(__name__).get_logger()


def cal_one_query(query, query_num, sample_group, sample_feature, r_list, q_list, table_dict, record_lock):
    gt_card, _ = o.query(query)
    rec_for_one_query = []
    est_card, _ = sample_group.query(query)
    t_num, row_num = gen_row_num(query, database)
    qh_vec = query_2_histogram_vec(query, database)
    query_vec = query_2_vec(query, database)
    query_sql = query_2_sql(query, table_dict)
    # query_vec['id_sql'] = query_sql
    query_vec['id_query_no'] = query_num
    sf = {}
    for sample_name, sample_card in est_card.items():
        record = {}
        record.update(query_vec)
        record.update(qh_vec)
        record['id_sample_name'] = sample_name
        for table_name, col_name in query.join:
            for temp_sample_name in sample_group.name_to_sample_unit[table_name].keys():
                total_name = f"{table_name}_{temp_sample_name}"
                if total_name in sample_name:
                    for k, v in sample_feature[table_name][temp_sample_name].items():
                        sf[f"{k}"] = v
        record.update(sf)
        record['label_GT_card'] = gt_card
        record['label_est_card'] = sample_card * row_num
        record['label_row_num'] = row_num
        rec_for_one_query.append(record)
    record_lock.acquire()
    r_list.extend(rec_for_one_query)
    q_list.append({"query_no": query_num, "query": query_sql, "cardinality_true": gt_card,
                  "use_table_num": t_num, "sel": gt_card/row_num})
    record_lock.release()


def gen_multi_dataset(dataset_name, workload_name, table_list, join_col, base_table, table_dict):
    data_path = DATA_ROOT / f"{dataset_name}"
    sql_path = WORKLOAD_ROOT / f"{dataset_name}" / f"{workload_name}.sql"
    global database
    database = Database(data_path, table_list, join_col, base_table)
    global o
    o = Oracle(database)
    # 1.Load workload
    queries = generate_workload_from_sql(sql_path, database)
    # 2.Load samples
    sg = load_sample_group(database, dataset_name)
    sf = load_sample_feature(sg, dataset_name)
    p_list = []
    lock = Lock()
    manager = Manager()
    record_list = manager.list()
    query_list = manager.list()
    temp_r_list = []
    temp_q_list = []
    L.info("Generating Start.")
    for i, q in enumerate(queries):
        p = Process(target=cal_one_query, args=(q, i, sg, sf, record_list, query_list, table_dict, lock,))
        p.start()
        p_list.append(p)
        if (i + 1) % 500 == 0:
            for p in p_list:
                p.join()
            p_list = []
        if (i + 1) % 1000 == 0:
            L.info(f"{i + 1} labels generated for {dataset_name}-{workload_name}")
        set_num = 10000
        if (i + 1) % set_num == 0:
            L.info(f"Saving Results of {i + 1} of {dataset_name}-{workload_name}")
            pkl_path = TEMP_DATASET_PATH / f"{dataset_name}-{workload_name}-{(i + 1)/set_num}.pkl"
            df = pd.DataFrame(list(record_list))
            df.to_pickle(pkl_path)
            csv_path = TEMP_DATASET_PATH / f"q-{dataset_name}-{workload_name}-{(i + 1)/set_num}.csv"
            df = pd.DataFrame(list(query_list))
            df.to_csv(csv_path)
            record_list[:] = []
            query_list[:] = []
            temp_r_list.append(pkl_path)
            temp_q_list.append(csv_path)
            break
    for p in p_list:
        p.join()
    csv_path = DATASET_PATH / f"{dataset_name}-{workload_name}.csv"
    pkl_path = DATASET_PATH / f"{dataset_name}-{workload_name}.pkl"
    df = pd.DataFrame(list(record_list))
    record_list[:] = []
    for r in temp_r_list:
        temp_df = pd.read_pickle(r)
        df = pd.concat([df, temp_df], axis=1)
    df.to_pickle(pkl_path)
    csv_path = DATASET_PATH / f"q-{dataset_name}-{workload_name}.csv"
    df = pd.DataFrame(list(query_list))
    query_list[:] = []
    for r in temp_q_list:
        temp_df = pd.read_csv(r)
        df = pd.concat([df, temp_df], axis=1)
    df.to_csv(csv_path)
