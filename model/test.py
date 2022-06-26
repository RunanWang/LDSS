import math
import time
import torch
import pandas as pd

import constants as C
from utils.log import Log, data_describe_log
from utils.metrics import cal_q_error
from model.feature import load_data, load_data_multi
from model.recommend import get_best_sample, get_optimal_best_sample


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
        q_error = cal_q_error(row_info)
        info = {"score": score, "ratio": row_info["ratio"], "GT_card": row_info["label_GT_card"],
                "est_card": row_info["label_est_card"], "q-error": q_error, "sample_name": row_info["id_sample_name"],
                "query_no": row_info["id_query_no"]}
        try:
            query_to_sample[row_info['id_query_no']][row_info["id_sample_name"]] = info
        except KeyError:
            query_to_sample[row_info['id_query_no']] = {}
            query_to_sample[row_info['id_query_no']][row_info["id_sample_name"]] = info
    return query_to_sample


def get_score_multi(model_m, loader, test_data, join_col):
    model_m.eval()
    output_value_list = []
    with torch.no_grad():
        for i, (wide_feat, deep_feat, sql_feat, co_feat, label) in enumerate(loader):
            output = model_m(wide_feat, deep_feat, sql_feat, co_feat)
            output_value_list += output.tolist()
    query_to_sample = {}
    for index, row in test_data.iterrows():
        row_info = dict(row)
        score_value = output_value_list[index]
        q_error = cal_q_error(row_info)
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
        info = {"score": score_value, "ratio": total_ratio, "GT_card": row_info["label_GT_card"],
                "est_card": row_info["label_est_card"], "q-error": q_error, "sample_name": row_info["id_sample_name"],
                "query_no": row_info["id_query_no"],  "use_cols": use_cols, "avg_ratio": avg_ratio,
                }
        try:
            query_to_sample[row_info["id_query_no"]][row_info["id_sample_name"]] = info
        except KeyError:
            query_to_sample[row_info["id_query_no"]] = {}
            query_to_sample[row_info["id_query_no"]][row_info["id_sample_name"]] = info
    return query_to_sample


def get_test_result(model_path, test_path, dataset_name, workload_name, table_col, join_col=None, multi=False, value=1.005):
    logger = Log(__name__, f"model-test-{dataset_name}").get_logger()
    logger.info("Loading model.")
    model = torch.load(model_path)
    test_df = pd.read_pickle(test_path)
    test_df = test_df.fillna(0)
    total_recommend_time = 0
    if multi:
        _, tloader, _, _ = load_data_multi(test_path, workload_name, logger,
                                           C.DEVICE, value, table_col, join_col, False)
        start_stmp = time.time()
        query_sample_to_info = get_score_multi(model, tloader, test_df, join_col)
        total_recommend_time = (time.time() - start_stmp) * 1e3
    else:
        _, tloader, _, _ = load_data(test_path, workload_name, logger, C.DEVICE, value, table_col, False)
        start_stmp = time.time()
        query_sample_to_info = get_score(model, tloader, test_df)
        total_recommend_time = (time.time() - start_stmp) * 1e3

    # Recommend
    query_to_best_sample_info = {}
    total_ratio = 0
    total_size = 0
    optimal_num = 0
    large_num = 0
    wrong_num = 0
    for query, sample_dict in query_sample_to_info.items():
        query_to_best_sample_info[query] = get_best_sample(sample_dict)
        q_b = get_best_sample(sample_dict)
        q_o = get_optimal_best_sample(sample_dict)
        total_size += 1
        total_ratio += query_to_best_sample_info[query]['ratio']
        if q_b["ratio"] <= q_o["ratio"] and q_b["q-error"] <= q_o["q-error"]:
            optimal_num += 1
        elif q_b["ratio"] > q_o["ratio"] and q_b["q-error"] <= q_o["q-error"]:
            large_num += 1
        elif q_b["ratio"] > q_o["ratio"] and q_b["q-error"] > q_o["q-error"] and q_b["q-error"] < value:
            large_num += 1
        elif q_b["ratio"] <= q_o["ratio"] and q_b["q-error"] > q_o["q-error"] and q_b["q-error"] < value:
            optimal_num += 1
        else:
            wrong_num += 1
    ratio_best = total_ratio / total_size
    optimal_ratio = optimal_num / total_size
    large_ratio = large_num / total_size
    wrong_ratio = wrong_num / total_size
    rec_time = total_recommend_time / total_size

    # Evaluate
    error_list = []
    for key, item in query_to_best_sample_info.items():
        error_list.append(item['q-error'])
    error_df = pd.DataFrame(error_list)
    data_describe_log(workload_name, error_df, logger)
    logger.info(f"Avg Sample Ratio={ratio_best:.4f}")
    logger.info(f"Optimal Ratio={optimal_ratio}, Large Ratio={large_ratio}, Wrong Ratio={wrong_ratio}")
    logger.info(f"Avg Recommend Time={rec_time:.2f}ms")
