from multiprocessing import Process, Queue
import pandas as pd
import time

import constants as C
from GenDataset.multi.dataset.table import Database
from GenDataset.multi.workload.gen_workload import generate_workload_from_sql
from GenDataset.multi.workload.workload import query_2_sql, query_2_vec, gen_row_num, query_2_histogram_vec
from GenDataset.multi.estimator.sample import load_sample_group, load_sample_feature, CorrelatedSampler
from GenDataset.multi.workload.gen_label import generate_label_by_standard_sample, generate_label_for_query
from GenDataset.multi.dataset.feat import load_histo
from utils.metrics import cal_q_error
from utils.log import Log

L = Log(__name__).get_logger()


def cal_vec(database, query, query_num, histo, standard_sample, sample_group, sample_feature, table_dict):
    # calculate true cardinality
    rec_for_one_query = []
    start_stmp = time.time()
    gt_card = generate_label_for_query(database, query)
    gt_time = (time.time() - start_stmp) / 60

    # cardinality estimation by sample group and standard sample
    start_stmp = time.time()
    t_num, row_num = gen_row_num(query, database)
    std_card, std_cost = standard_sample.query(query)
    std_card = std_card * row_num
    standard_q_error = cal_q_error(gt_card, std_card)
    card, cost = sample_group.query(query)
    sg_time = (time.time() - start_stmp) / 60

    # gnenrate features for query
    start_stmp = time.time()
    histo_feat = query_2_histogram_vec(query, histo)
    query_vec = query_2_vec(query, database)
    query_sql = query_2_sql(query, table_dict)

    query_vec.update(histo_feat)
    query_vec['id_query_no'] = query_num
    query_vec['label_GT_card'] = gt_card
    query_vec['label_std_qerror'] = standard_q_error
    query_vec['label_row_num'] = row_num
    query_vec['label_std_time'] = std_cost
    qv_time = (time.time() - start_stmp) / 60

    # packing query and sample together
    start_stmp = time.time()
    for sample_name, sample_card in card.items():
        sample_card = sample_card * row_num
        record = {}
        record.update(query_vec)
        record['id_sample_name'] = sample_name
        # for table_name, col_name in query.join:
        #     for temp_sample_name in sample_group.name_to_sample_unit[table_name].keys():
        #         total_name = f"{table_name}_{temp_sample_name}"
        #         if total_name in sample_name:
        #             for k, v in sample_feature[table_name][temp_sample_name].items():
        #                 sf[f"{k}"] = v
        # record.update(sf)
        record.update(sample_feature[sample_name])
        sample_q_error = cal_q_error(gt_card, sample_card)
        record['label'] = generate_label_by_standard_sample(standard_q_error, sample_q_error)
        record['label_q_error'] = sample_q_error
        record['label_est_card'] = sample_card
        record['label_cost_time'] = cost[sample_name]
        rec_for_one_query.append(record)
    packing_time = (time.time() - start_stmp) / 60
    query_info = {"query_no": query_num, "query": query_sql, "use_table_num": t_num, "cardinality_true": gt_card, "sel": gt_card/row_num}
    time_info = {"gt_time": gt_time, "sg_time": sg_time, "qv_time": qv_time, "packing_time": packing_time}
    return rec_for_one_query, query_info, time_info


def cal_query_set(id, queries, query_num_offset, database, histo, standard_sample, sample_group, sample_feature, table_dict, q: Queue):
    rec_for_queryset = []
    query_info_list = []
    time_dict = {"gt_time": 0, "sg_time": 0, "qv_time": 0, "packing_time": 0}
    for i, query in enumerate(queries):
        rec_for_one_query, query_info, time_info = cal_vec(
            database, query, query_num_offset + i, histo, standard_sample, sample_group, sample_feature, table_dict)
        rec_for_queryset += rec_for_one_query
        query_info_list.append(query_info)
        for k, v in time_dict.items():
            time_dict[k] = v + time_info[k]
        if (i + 1) % 1000 == 0:
            L.info(f"Query of group {id:>2d} - {(i + 1):>6d} / {len(queries)}")
    q.put((rec_for_queryset, query_info_list, time_dict))


def gen_multi_dataset(dataset_name, workload_name, table_list, join_col, base_table, table_dict):
    timer_dist = 0
    timer_fence = 0
    timer_save = 0
    mp_num = 64

    # Load database
    data_path = C.DATA_ROOT / f"{dataset_name}"
    database = Database(data_path, table_list, join_col, base_table)

    # Load workload
    sql_path = C.WORKLOAD_ROOT / f"{dataset_name}" / f"{workload_name}.sql"
    queries = generate_workload_from_sql(sql_path, database)

    # Load samples
    sampling_group = load_sample_group(database, dataset_name)
    histo = {}
    total_df = {}
    ref_df = {}
    standard_sample = CorrelatedSampler(database, C.STANDARD_MULTI_SAMPLE_PAR["ratio"], C.STANDARD_MULTI_SAMPLE_PAR["seed"])
    for table_name, table in database.table.items():
        total_df[table_name] = table.data
        ref_df[table_name] = standard_sample.sample[table_name]
        histo[table_name] = load_histo(f"{dataset_name}.{table_name}", table.data)
    sample_features = load_sample_feature(sampling_group, dataset_name, total_df, ref_df)

    q = Queue()
    start_stmp_total = time.time()
    start_stmp = time.time()
    L.info(f"{workload_name} is being processed!")
    p_list = []
    batch_size = int(len(queries) / mp_num)
    for i in range(0, mp_num):
        start_id = i * batch_size
        end_id = (i + 1) * batch_size
        if i == (mp_num - 1):
            end_id = None
        queries_in_mp = queries[start_id:end_id]
        p = Process(target=cal_query_set, args=(i, queries_in_mp, start_id, database, histo, standard_sample, sampling_group, sample_features, table_dict, q,))
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
    L.info("Saving records")
    start_stmp = time.time()
    pkl_path = C.DATASET_PATH / f"{dataset_name}-{workload_name}.pkl"
    csv_path = C.DATASET_PATH / f"{dataset_name}-{workload_name}.csv"
    df = pd.DataFrame(total_rec_list)
    timer_df = (time.time() - start_stmp) / 60
    stmp_d = time.time()
    df.to_pickle(pkl_path)
    # df.to_csv(csv_path)
    timer_todisk = (time.time() - stmp_d) / 60
    csv_path = C.DATASET_PATH / f"q-{dataset_name}-{workload_name}.csv"
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
