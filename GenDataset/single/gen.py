import pandas as pd
import time
from multiprocessing import Process, Manager, Lock, Queue
from GenDataset.single.dataset.feat import load_histo

from utils.log import Log
from utils.metrics import cal_q_error
from constants import DATASET_PATH, STANDARD_SAMPLE_PAR
from GenDataset.single.workload.gen_label import generate_label_for_query, generate_label_by_standard_sample
from GenDataset.single.dataset.dataset import load_table
from GenDataset.single.workload.workload import load_gen_queryset, query_2_vec, query_2_sql, query_2_histogram_vec
from GenDataset.single.estimator.sample_group import load_sample_feature, load_sample_group
from GenDataset.single.estimator.sample import Sampling

L = Log(__name__).get_logger()


def cal_query_set(id, queries, query_num_offset, table, histo, standard_sample, sample_group, sample_feature, q: Queue):
    rec_for_queryset = []
    query_info_list = []
    time_dict = {"gt_time": 0, "sg_time": 0, "qv_time": 0, "packing_time": 0}
    for i, query in enumerate(queries):
        rec_for_one_query, query_info, time_info = cal_vec(
            table, query, query_num_offset + i, histo, standard_sample, sample_group, sample_feature)
        rec_for_queryset += rec_for_one_query
        query_info_list.append(query_info)
        for k, v in time_dict.items():
            time_dict[k] = v + time_info[k]
        if (i + 1) % 1000 == 0:
            L.info(f"Query of group {id:>2d} - {(i + 1):>6d} / {len(queries)}")
    q.put((rec_for_queryset, query_info_list, time_dict))


def cal_vec(table, query, query_num, histo, standard_sample, sample_group, sample_feature):
    # calculate true cardinality
    rec_for_one_query = []
    start_stmp = time.time()
    gt_card = generate_label_for_query(table, query)
    gt_time = (time.time() - start_stmp) / 60

    # cardinality estimation by sample group and standard sample
    start_stmp = time.time()
    std_card, std_cost = standard_sample.query2(query)
    standard_q_error = cal_q_error(gt_card, std_card)
    card, cost = sample_group.query2(query)
    sg_time = (time.time() - start_stmp) / 60

    # gnenrate features for query
    start_stmp = time.time()
    histo_feat = query_2_histogram_vec(query, histo)
    query_vec = query_2_vec(query, table)
    query_sql = query_2_sql(query, table, aggregate=False, dbms='postgres')
    # query_vec['id_sql'] = query_sql
    query_vec.update(histo_feat)
    query_vec['id_query_no'] = query_num
    query_vec['label_GT_card'] = gt_card
    query_vec['label_std_qerror'] = standard_q_error
    query_vec['label_row_num'] = table.row_num
    query_vec['label_std_time'] = std_cost
    qv_time = (time.time() - start_stmp) / 60

    # packing query and sample together
    start_stmp = time.time()
    for sample_name, sample_card in card.items():
        record = {}
        record.update(query_vec)
        record['id_sample_name'] = sample_name
        record.update(sample_feature[sample_name])
        sample_q_error = cal_q_error(gt_card, sample_card)
        record['label'] = generate_label_by_standard_sample(standard_q_error, sample_q_error)
        record['label_q_error'] = sample_q_error
        record['label_est_card'] = sample_card
        record['label_cost_time'] = cost[sample_name]
        rec_for_one_query.append(record)
    packing_time = (time.time() - start_stmp) / 60
    query_info = {"query_no": query_num, "query": query_sql, "cardinality_true": gt_card}
    time_info = {"gt_time": gt_time, "sg_time": sg_time, "qv_time": qv_time, "packing_time": packing_time}
    return rec_for_one_query, query_info, time_info


def cal_one_query(t, query, query_num, sample_group, standard_sample, sample_feature, histo, r_list, q_list, t_list, record_lock):
    rec_for_one_query, query_info, time_info = cal_vec(
        t, query, query_num, histo, standard_sample, sample_group, sample_feature)
    start_stmp = time.time()
    record_lock.acquire()
    r_list.extend(rec_for_one_query)
    q_list.append(query_info)
    locking_time = (time.time() - start_stmp) / 60
    time_info["locking_time"] = locking_time
    t_list.append(time_info)
    record_lock.release()


def gen_single_dataset(dataset_name, workload_name):
    timer_dist = 0
    timer_fence = 0
    timer_save = 0
    time_df = pd.DataFrame()
    # Load workload
    queryset = load_gen_queryset(dataset_name)

    # Load table
    table = load_table(dataset_name)

    # Load samples
    std_sample_ratio = STANDARD_SAMPLE_PAR["ratio"]
    standard_sample = Sampling(table, std_sample_ratio, STANDARD_SAMPLE_PAR["seed"])
    sampling_group = load_sample_group(table, dataset_name)
    sample_features = load_sample_feature(sampling_group, dataset_name, table.data, standard_sample.sample)
    histo = load_histo(dataset_name, table.data)
    start_stmp_total = time.time()
    for group, queries in queryset.items():
        L.info(group + " group is being processed!")
        p_list = []
        lock = Lock()
        manager = Manager()
        record_list = manager.list()
        query_list = manager.list()
        time_list = manager.list()
        for i, q in enumerate(queries):
            start_stmp = time.time()
            p = Process(target=cal_one_query,
                        args=(table, q, i, sampling_group, standard_sample, sample_features, histo, record_list, query_list, time_list, lock,))
            p.start()
            p_list.append(p)
            timer_dist += (time.time() - start_stmp) / 60
            start_stmp = time.time()
            if (i + 1) % 32 == 0:
                for p in p_list:
                    p.join()
            if (i + 1) % 1000 == 0:
                L.info("Query of group " + group + " - {:>6d} / {}".format((i + 1), len(queries)))
            timer_fence += (time.time() - start_stmp) / 60
        L.info("Saving records of group " + group)
        start_stmp = time.time()
        pkl_path = DATASET_PATH / f"{dataset_name}-{workload_name}-{group}.pkl"
        df = pd.DataFrame(list(record_list))
        record_list[:] = []
        df.to_pickle(pkl_path)
        csv_path = DATASET_PATH / f"q-{dataset_name}-{workload_name}.csv"
        df = pd.DataFrame(list(query_list))
        query_list[:] = []
        df.to_csv(csv_path)
        df = pd.DataFrame(list(time_list))
        timer_save += (time.time() - start_stmp) / 60
        time_df = pd.concat([time_df, df])
        time_list[:] = []
    timer_total = (time.time() - start_stmp_total) / 60
    L.info("Generating Dataset Time Cost Summary as Follows:")
    L.info(f"Main Thread Total time {timer_total} min.")
    L.info(f"Distribute Process time {timer_dist} min. {timer_dist / timer_total * 100}% of total time.")
    L.info(f"Waiting Process time {timer_fence} min. {timer_fence / timer_total * 100}% of total time.")
    L.info(f"Saving Records time {timer_save} min. {timer_save / timer_total * 100}% of total time.")
    gt_time = time_df["gt_time"].sum()
    sg_time = time_df["sg_time"].sum()
    qv_time = time_df["qv_time"].sum()
    packing_time = time_df["packing_time"].sum()
    locking_time = time_df["locking_time"].sum()
    total_cal_time = gt_time + sg_time + qv_time + packing_time + locking_time
    L.info("Calculating Time as Follows:")
    L.info(f"Total cal time {total_cal_time} min. {total_cal_time / timer_total * 100}% of total time.")
    L.info(f"Ground Turth time {gt_time} min. {gt_time / total_cal_time * 100}% of total cal time.")
    L.info(f"Sample Group time {sg_time} min. {sg_time / total_cal_time * 100}% of total cal time.")
    L.info(f"Query Vector time {qv_time} min. {qv_time / total_cal_time * 100}% of total cal time.")
    L.info(f"Packing time {packing_time} min. {packing_time / total_cal_time * 100}% of total cal time.")
    L.info(f"Locking time {locking_time} min. {locking_time / total_cal_time * 100}% of total cal time.")


def gen_single_dataset_batch(dataset_name, workload_name):
    timer_dist = 0
    timer_fence = 0
    timer_save = 0
    mp_num = 32
    # Load workload
    queryset = load_gen_queryset(dataset_name)

    # Load table
    table = load_table(dataset_name)

    # Load samples
    std_sample_ratio = STANDARD_SAMPLE_PAR["ratio"]
    standard_sample = Sampling(table, std_sample_ratio, STANDARD_SAMPLE_PAR["seed"])
    sampling_group = load_sample_group(table, dataset_name)
    sample_features = load_sample_feature(sampling_group, dataset_name, table.data, standard_sample.sample)
    histo = load_histo(dataset_name, table.data)

    q = Queue()
    start_stmp_total = time.time()
    for group, queries in queryset.items():
        if group == "train" or group == "valid":
            continue
        start_stmp = time.time()
        L.info(group + " group is being processed!")
        p_list = []
        batch_size = int(len(queries) / mp_num)
        for i in range(0, mp_num):
            start_id = i * batch_size
            end_id = (i + 1) * batch_size
            if i == (mp_num - 1):
                end_id = None
            queries_in_mp = queries[start_id:end_id]
            p = Process(target=cal_query_set,
                        args=(i, queries_in_mp, start_id, table, histo, standard_sample, sampling_group, sample_features, q,))
            p.start()
            p_list.append(p)
        timer_dist += (time.time() - start_stmp) / 60
        L.info("All batch ditributed!")
        start_stmp = time.time()
        total_rec_list = []
        total_query_info_list = []
        total_time_dict = {"gt_time": 0, "sg_time": 0, "qv_time": 0, "packing_time": 0}
        for p in p_list:
            rec_for_queryset, query_info_list, time_dict = q.get()
            total_rec_list += rec_for_queryset
            total_query_info_list += query_info_list
            for k, v in total_time_dict.items():
                total_time_dict[k] = v + time_dict[k]
        for p in p_list:
            p.join()
        timer_fence += (time.time() - start_stmp) / 60
        L.info("Saving records of group " + group)
        start_stmp = time.time()
        pkl_path = DATASET_PATH / f"{dataset_name}-{workload_name}-{group}.pkl"
        df = pd.DataFrame(total_rec_list)
        timer_df = (time.time() - start_stmp) / 60
        stmp_d = time.time()
        df.to_pickle(pkl_path)
        timer_todisk = (time.time() - stmp_d) / 60
        csv_path = DATASET_PATH / f"q-{dataset_name}-{workload_name}.csv"
        df = pd.DataFrame(total_query_info_list)
        df.to_csv(csv_path)
        timer_save += (time.time() - start_stmp) / 60
    timer_total = (time.time() - start_stmp_total) / 60
    L.info("Generating Dataset Time Cost Summary as Follows:")
    L.info(f"Main Thread Total time {timer_total} min.")
    L.info(f"Distribute Process time {timer_dist} min. {timer_dist / timer_total * 100}% of total time.")
    L.info(f"Waiting Process time {timer_fence} min. {timer_fence / timer_total * 100}% of total time.")
    L.info(f"Saving Records time {timer_save} min. {timer_save / timer_total * 100}% of total time.")
    L.info(f"Df time {timer_df} min. {timer_df / timer_save * 100}% of total time.")
    L.info(f"ToDisk time {timer_todisk} min. {timer_todisk / timer_save * 100}% of total time.")
    gt_time = total_time_dict["gt_time"]
    sg_time = total_time_dict["sg_time"]
    qv_time = total_time_dict["qv_time"]
    packing_time = total_time_dict["packing_time"]
    total_cal_time = gt_time + sg_time + qv_time + packing_time
    L.info("Calculating Time as Follows:")
    L.info(f"Total cal time {total_cal_time} min. {total_cal_time / timer_total * 100}% of total time.")
    L.info(f"Ground Turth time {gt_time} min. {gt_time / total_cal_time * 100}% of total cal time.")
    L.info(f"Sample Group time {sg_time} min. {sg_time / total_cal_time * 100}% of total cal time.")
    L.info(f"Query Vector time {qv_time} min. {qv_time / total_cal_time * 100}% of total cal time.")
    L.info(f"Packing time {packing_time} min. {packing_time / total_cal_time * 100}% of total cal time.")
