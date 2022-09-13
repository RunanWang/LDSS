import math
import time
import torch
import lightgbm as lgb
import pandas as pd

import constants as C
from utils.log import Log, data_describe_log
from utils.metrics import cal_q_error
from model.featureNew import load_data, load_data_multi, load_data_gbdt, load_data_multi_gbdt
import model.recommend as R
from GenDataset.single.dataset.dataset import load_table
from GenDataset.single.workload.workload import load_gen_queryset
from GenDataset.multi.workload.gen_workload import generate_workload_from_sql
from GenDataset.single.estimator.sample import Sampling
from GenDataset.multi.dataset.table import Database
from GenDataset.multi.estimator.sample import CorrelatedSampler2
from GenDataset.multi.workload.workload import gen_row_num


def get_score(model_m, loader, test_data):
    model_m.eval()
    output_list = []
    with torch.no_grad():
        for i, (wide_feat, deep_feat, sql_feat, co_feat, label) in enumerate(loader):
            output = model_m(wide_feat, deep_feat, sql_feat, co_feat)
            output_list += output.tolist()
    query_to_sample = {}
    for index, row in test_data.iterrows():
        row_info = dict(row)
        score = output_list[index]
        q_error = cal_q_error(row_info['label_GT_card'], row_info['label_est_card'])
        info = {"score": score, "ratio": row_info["ratio"], "GT_card": row_info["label_GT_card"],
                "est_card": row_info["label_est_card"], "q-error": q_error, "sample_name": row_info["id_sample_name"],
                "query_no": row_info["id_query_no"], "std_qerror": row_info["std_qerror"]}
        try:
            query_to_sample[row_info['id_query_no']][row_info["id_sample_name"]] = info
        except KeyError:
            query_to_sample[row_info['id_query_no']] = {}
            query_to_sample[row_info['id_query_no']][row_info["id_sample_name"]] = info
    return query_to_sample


def get_score_new(model_m, loader, test_data):
    model_m.eval()
    output_list = []
    with torch.no_grad():
        for _, (wide_feat, deep_feat, co_feat, sql_feat, _) in enumerate(loader):
            output = model_m(wide_feat, deep_feat, co_feat, sql_feat)
            output_list += output.tolist()
    query_to_sample = {}
    for index, row in test_data.iterrows():
        row_info = dict(row)
        score = output_list[index]
        info = {"score": score, "ratio": row_info["ratio"], "q-error": row_info["label_q_error"],
                "std_qerror": row_info["label_std_qerror"], "time": row_info["label_cost_time"],
                "std_time": row_info["label_std_time"], "sample_name": row_info["id_sample_name"],
                "GT_card": row_info["label_GT_card"], "est_card": row_info["label_est_card"],
                "label": row_info["label"]}
        try:
            query_to_sample[row_info['id_query_no']][row_info["id_sample_name"]] = info
        except KeyError:
            query_to_sample[row_info['id_query_no']] = {}
            query_to_sample[row_info['id_query_no']][row_info["id_sample_name"]] = info
    return query_to_sample


def get_score_gdbt(eval_list, test_data):
    query_to_sample = {}
    for index, row in test_data.iterrows():
        row_info = dict(row)
        score = eval_list[index]
        info = {"score": [score], "ratio": row_info["ratio"], "q-error": row_info["label_q_error"],
                "std_qerror": row_info["label_std_qerror"], "time": row_info["label_cost_time"],
                "std_time": row_info["label_std_time"], "sample_name": row_info["id_sample_name"],
                "GT_card": row_info["label_GT_card"], "est_card": row_info["label_est_card"],
                "label": row_info["label"]}
        try:
            query_to_sample[row_info['id_query_no']][row_info["id_sample_name"]] = info
        except KeyError:
            query_to_sample[row_info['id_query_no']] = {}
            query_to_sample[row_info['id_query_no']][row_info["id_sample_name"]] = info
    return query_to_sample


def get_score_multi(model_m, loader, test_data):
    model_m.eval()
    output_value_list = []
    st_time = time.time()
    with torch.no_grad():
        for i, (wide_feat, deep_feat, sql_feat, co_feat, label) in enumerate(loader):
            output = model_m(wide_feat, deep_feat, sql_feat, co_feat)
            output_value_list += output.tolist()
    query_to_sample = {}
    for index, row in test_data.iterrows():
        row_info = dict(row)
        score = output_value_list[index]
        total_ratio = 1
        total_ratio = row_info["of_ratio"]
        info = {"score": score, "ratio": total_ratio, "q-error": row_info["label_q_error"],
                "std_qerror": row_info["label_std_qerror"], "time": row_info["label_cost_time"],
                "std_time": row_info["label_std_time"], "sample_name": row_info["id_sample_name"],
                "GT_card": row_info["label_GT_card"], "est_card": row_info["label_est_card"],
                "label": row_info["label"]}
        try:
            query_to_sample[row_info["id_query_no"]][row_info["id_sample_name"]] = info
        except KeyError:
            query_to_sample[row_info["id_query_no"]] = {}
            query_to_sample[row_info["id_query_no"]][row_info["id_sample_name"]] = info
    total_time = (time.time() - st_time) * 1e-3
    return query_to_sample, total_time


def get_score_multi_gbdt(eval_list, test_data, join_col):
    query_to_sample = {}
    for index, row in test_data.iterrows():
        row_info = dict(row)
        score = eval_list[index]
        total_ratio = 1
        use_cols = 0
        for table_name, col_name in join_col.items():
            use_cols = row_info[f"ut_{table_name}_{col_name}"] + use_cols
            if row_info[f"of_{table_name}_ratio"] != 0:
                total_ratio *= row_info[f"of_{table_name}_ratio"]
        avg_ratio = math.pow(total_ratio, 1/use_cols)
        if "title_hs_title" in row_info["id_sample_name"]:
            total_ratio = row_info["of_title_ratio"]
            avg_ratio = total_ratio
        info = {"score": [score], "ratio": total_ratio, "q-error": row_info["label_q_error"],
                "std_qerror": row_info["label_std_qerror"], "time": row_info["label_cost_time"],
                "std_time": row_info["label_std_time"], "sample_name": row_info["id_sample_name"],
                "GT_card": row_info["label_GT_card"], "est_card": row_info["label_est_card"],
                "label": row_info["label"], "avg_ratio": avg_ratio}
        try:
            query_to_sample[row_info["id_query_no"]][row_info["id_sample_name"]] = info
        except KeyError:
            query_to_sample[row_info["id_query_no"]] = {}
            query_to_sample[row_info["id_query_no"]][row_info["id_sample_name"]] = info
    return query_to_sample


def get_rec_info(query_sample_to_info, rec_method, logger):
    query_to_best_sample_info = {}
    sample_name_to_time = {}
    total_ratio = 0
    total_size = 0
    sample_time_total = 0
    std_time_total = 0
    total_recommend_time = 0
    start_stmp = time.time()
    for query, sample_dict in query_sample_to_info.items():
        query_to_best_sample_info[query] = rec_method(sample_dict)
        total_size += 1
        total_ratio += query_to_best_sample_info[query]['ratio']
        sample_time_total += query_to_best_sample_info[query]['time']
        std_time_total += query_to_best_sample_info[query]['std_time']
        sample_name = query_to_best_sample_info[query]["sample_name"]
        # logger.info(query_to_best_sample_info[query])
        try:
            sample_name_to_time[sample_name] += 1
        except:
            sample_name_to_time[sample_name] = 1
    logger.info(sample_name_to_time)
    total_recommend_time += (time.time() - start_stmp) * 1e3
    ratio_best = total_ratio / total_size
    rec_time = total_recommend_time
    sample_time = sample_time_total / total_size
    std_time = std_time_total / total_size
    error_list = []
    error_std_list = []
    for key, item in query_to_best_sample_info.items():
        error_list.append(item['q-error'])
        error_std_list.append(item['std_qerror'])
    return ratio_best, rec_time, sample_time, std_time, error_list, error_std_list


def get_test_result(model_path, test_path, dataset_name, workload_name, table_col, join_col=None, model_type="nn", multi=False, test_same_size=False):
    logger = Log(__name__, f"model-test-{dataset_name}").get_logger()
    logger.info("Loading model.")
    if model_type == "nn":
        model = torch.load(model_path)
    else:
        model = lgb.Booster(model_file=str(model_path))
    logger.info("Loading Data.")
    test_df = pd.read_pickle(test_path)
    test_df = test_df.fillna(0)

    logger.info("Calculating Score from model.")
    if multi:
        cols = ["of_ratio", "id_query_no", "id_sample_name", "label_GT_card", "label_est_card",
                "label_std_qerror", "label_q_error", "label_cost_time", "label_std_time", "label"]
        for table_name, col_name in join_col.items():
            cols.append(f"ut_{table_name}_{col_name}")
        test_df = test_df[cols]
        if model_type == "nn":
            _, tloader, _, _ = load_data_multi(test_path, workload_name, logger, C.DEVICE, table_col, join_col, False)
            query_sample_to_info, total_recommend_time = get_score_multi(model, tloader, test_df)
        else:
            test_X, _ = load_data_multi_gbdt(test_path, "test", logger, table_col, join_col)
            pred_y = model.predict(test_X)
            query_sample_to_info = get_score_multi_gbdt(pred_y, test_df, join_col)
    else:
        test_df = test_df[["ratio", "id_query_no", "id_sample_name", "label_GT_card", "label_est_card",
                           "label_std_qerror", "label_q_error", "label_cost_time", "label_std_time", "label"]]
        if model_type == "nn":
            _, tloader, _, _ = load_data(test_path, workload_name, logger, C.DEVICE, table_col, False)
            start_stmp = time.time()
            query_sample_to_info = get_score_new(model, tloader, test_df)
            total_recommend_time = (time.time() - start_stmp) * 1e3
        else:
            test_X, _ = load_data_gbdt(test_path, "test", logger, table_col)
            start_stmp = time.time()
            pred_y = model.predict(test_X)
            query_sample_to_info = get_score_gdbt(pred_y, test_df)
            total_recommend_time = (time.time() - start_stmp) * 1e3

    # Recommend
    logger.info("Recommending sample from Meta-info.")
    ratio_ldss, rec_time_ldss, sample_time_ldss, std_time, error_ldss, error_std_list = get_rec_info(query_sample_to_info, R.smallest_sample_wins, logger)
    total_recommend_time = (total_recommend_time + rec_time_ldss) / len(error_ldss)

    # Evaluate
    logger.info("Error of CE using Standard Sample")
    error_df_2 = pd.DataFrame(error_std_list)
    data_describe_log(workload_name, error_df_2, logger)
    logger.info(f"Avg Sample Time={std_time:.2f}ms")

    logger.info("Error of CE using LDSS")
    error_df = pd.DataFrame(error_ldss)
    data_describe_log(workload_name, error_df, logger)
    logger.info(f"LDSS Avg Recommend Time={total_recommend_time:.2f}ms, Avg Sample Time={sample_time_ldss:.2f}ms")
    logger.info(f"Avg Sample Ratio={ratio_ldss:.4f}")

    cal_three_ratio(query_sample_to_info, R.smallest_sample_wins, logger)

    # Same Size Sampler
    if test_same_size:
        ratio_opt, _, sample_time_opt, _, error_opt, _ = get_rec_info(query_sample_to_info, R.opt_sample_wins, logger)
        logger.info("Error of CE using Optimal Sample")
        error_df_2 = pd.DataFrame(error_opt)
        data_describe_log(workload_name, error_df_2, logger)
        logger.info(f"Avg Sample Time={sample_time_opt:.2f}ms")
        logger.info(f"Avg Sample Ratio={ratio_opt:.4f}")
        if multi:
            multi_get_same_size_sampler_result(dataset_name, workload_name, ratio_ldss, logger)
        else:
            single_get_same_size_sampler_result(dataset_name, workload_name, ratio_ldss, logger)


def single_get_same_size_sampler_result(dataset_name, workload_name, sample_ratio, logger):
    error_list = []
    queryset = load_gen_queryset(dataset_name)
    table = load_table(dataset_name)
    same_size_sampler = Sampling(table, sample_ratio, C.STANDARD_SAMPLE_PAR["seed"])
    csv_path = C.DATASET_PATH / f"q-{dataset_name}-generate.csv"
    card_df = pd.read_csv(csv_path, index_col=0)
    query_no = 0
    sample_time = 0
    for query in queryset[workload_name]:
        std_card, cost_time = same_size_sampler.query2(query)
        gt_card = card_df[card_df["query_no"] == query_no]["cardinality_true"].to_list()[0]
        query_no += 1
        error_list.append(cal_q_error(gt_card, std_card))
        sample_time += cost_time

    logger.info("Error of CE using Same Size Sampler")
    error_df = pd.DataFrame(error_list)
    data_describe_log(workload_name, error_df, logger)
    logger.info(f"Avg Sample Time={sample_time / query_no:.2f}ms")


def multi_get_same_size_sampler_result(dataset_name, workload_name, sample_ratio, logger):
    error_list = []
    sql_path = C.WORKLOAD_ROOT / f"{dataset_name}" / f"{workload_name}.sql"
    data_path = C.DATA_ROOT / f"{dataset_name}"

    database = Database(data_path, C.JOB_TABLE_LIST, C.JOB_JOIN_COL, C.JOB_BASE_TABLE)
    queries = generate_workload_from_sql(sql_path, database)

    same_size_sampler = CorrelatedSampler2(database, sample_ratio, C.STANDARD_MULTI_SAMPLE_PAR["seed"])
    csv_path = C.DATASET_PATH / f"q-{dataset_name}-{workload_name}.csv"
    card_df = pd.read_csv(csv_path, index_col=0)
    query_no = 0
    sample_time = 0
    for query in queries:
        t_num, row_num = gen_row_num(query, database)
        std_card, cost_time = same_size_sampler.query(query)
        std_card = std_card * row_num
        gt_card = card_df[card_df["query_no"] == query_no]["cardinality_true"].to_list()[0]
        query_no += 1
        error_list.append(cal_q_error(gt_card, std_card))
        sample_time += cost_time

    logger.info("Error of CE using Same Size Sampler")
    error_df = pd.DataFrame(error_list)
    data_describe_log(workload_name, error_df, logger)
    logger.info(f"Avg Sample Time={sample_time / query_no:.2f}ms")


def cal_three_ratio(query_sample_to_info, method, logger):
    total_size = 0
    optimal_num = 0
    large_num = 0
    wrong_num = 0
    for query, sample_dict in query_sample_to_info.items():
        q_b = method(sample_dict)
        q_s = R.standard_sample_wins(sample_dict)
        has_sample = R.has_small_sample(sample_dict)
        total_size += 1
        if q_b["q-error"] > q_s["q-error"]:
            wrong_num += 1
        elif q_b["ratio"] < q_s["ratio"]:
            optimal_num += 1
        elif has_sample and q_b["ratio"] >= q_s["ratio"]:
            large_num += 1
        else:
            optimal_num += 1
    optimal_ratio = optimal_num / total_size
    large_ratio = large_num / total_size
    wrong_ratio = wrong_num / total_size
    logger.info(f"opt={round(optimal_ratio, 2)}, large={round(large_ratio, 2)}, wrong={round(wrong_ratio, 2)}")
