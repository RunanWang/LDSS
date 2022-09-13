from copy import deepcopy
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import constants as C
from utils.log import Log

COLS = ["Record_Type", "Registration_Class", "State", "County", "Body_Type", "Fuel_Type",
        "Reg_Valid_Date", "Color", "Scofflaw_Indicator", "Suspension_Indicator", "Revocation_Indicator"]

L = Log(__name__, "update").get_logger()


def prepare_data(data):
    data = deepcopy(data)
    data["sel"] = data["cardinality_true"] / 11591877 + 1 / 11591877
    data["sel"] = np.log10(data["sel"])
    data["est_std"] = data["est_std"] / 11591877 + 1 / 11591877
    data["std"] = np.log10(data["est_std"])
    data["est_s0.01"] = data["est_s0.01"] / 11591877 + 1 / 11591877
    data["s0.01"] = np.log10(data["est_s0.01"])
    data["est_s0.05"] = data["est_s0.05"] / 11591877 + 1 / 11591877
    data["s0.05"] = np.log10(data["est_s0.05"])
    use_col = []
    total_col = []
    to_drop = ["cardinality_true", "Unnamed: 0", "query", "query_no"]
    for col in COLS:
        col_name = col + "_total"
        data[col_name] = data[col + "_qr"] - data[col + "_ql"]
        def f(row): return - row[col + "_qr"] if row[col_name] == 0 else row[col_name]
        data[col_name] = data.apply(f, axis=1)
    for col in data.columns:
        if "_use" in col:
            use_col.append(col)
            to_drop.append(col)
        elif "_ql" in col:
            to_drop.append(col)
        elif "_qr" in col:
            to_drop.append(col)
        elif "_total" in col:
            total_col.append(col)
            to_drop.append(col)
        elif "est_" in col:
            to_drop.append(col)

    X = data.drop(to_drop, axis=1)
    # L.info(X.head(5))
    data["use_total"] = 0
    for col in use_col:
        data["use_total"] = data["use_total"] + data[col]
    to_drop.append("use_total")
    data["sel"] = (data["sel"] - data["sel"].min()) / (data["sel"].max() - data["sel"].min())
    data["use_total"] = (data["use_total"] - data["use_total"].min()) / (data["use_total"].max() - data["use_total"].min())
    Y = data[["use_total", "sel"]]
    use_df = data[use_col]
    X_v = X.values
    Y_v = Y.values
    sql_df = data[["query", "query_no"]]
    return X_v, Y_v, use_df, sql_df


def get_cluster1(X_v):
    clustering = DBSCAN(eps=1, min_samples=10, metric="manhattan").fit(X_v)
    labels = clustering.labels_
    df = pd.DataFrame(labels)
    df.to_csv(C.DATASET_PATH / "update" / "cluster-labels.csv")
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    L.info("Estimated number of clusters: %d" % n_clusters_)
    L.info("Estimated number of noise points: %d" % n_noise_)
    return clustering


def get_cluster2(X_v):
    clustering = DBSCAN(eps=0.99, min_samples=10, metric="manhattan").fit(X_v)
    labels = clustering.labels_
    df = pd.DataFrame(labels)
    df.to_csv(C.DATASET_PATH / "update" / "cluster-labels2.csv")
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    L.info("Estimated number of clusters: %d" % n_clusters_)
    L.info("Estimated number of noise points: %d" % n_noise_)
    return clustering


def get_small_queryset(dataset_name):
    sample_num = 1000
    data = pd.read_csv(C.DATASET_PATH / "update" / f"q-{dataset_name}.csv")
    X_v, Y_v, use_df, sql_df = prepare_data(data)
    clustering1 = get_cluster1(X_v)
    # clustering2 = get_cluster2(Y_v)
    labels1 = clustering1.labels_

    # labels2 = clustering2.labels_
    n_clusters_1 = len(set(labels1)) - (1 if -1 in labels1 else 0)

    core_samples_mask = np.zeros_like(labels1, dtype=bool)
    core_samples_mask[clustering1.core_sample_indices_] = True
    df_core = pd.DataFrame(core_samples_mask, columns=["core"])
    df1 = pd.DataFrame(labels1, columns=["cluster1"])
    df = pd.concat([use_df, df1, df_core, sql_df], axis=1)
    cluster_2_num = {}
    df_core_queries = df[(df["core"] == True)]
    for i in range(-1, n_clusters_1):
        cluster_2_num[i] = int(len(df[(df["cluster1"] == i) & (df["core"] == True)]) / len(df_core_queries) * sample_num) + 1
    # df2 = pd.DataFrame(labels2, columns=["cluster2"])
    total_core_num = 0
    cluster_2_num[-1] = 0
    for core, core_num in cluster_2_num.items():
        total_core_num += core_num
    cluster_2_num[0] = cluster_2_num[0] - total_core_num + sample_num
    df_total = pd.DataFrame()
    for core, core_num in cluster_2_num.items():
        L.info(f"Sampling {core_num} items for {core}")
        each_col = int(core_num/(len(COLS)))
        remain_num = core_num - each_col * (len(COLS))
        if core_num > len(COLS):
            i = 0
            for col in COLS:
                df_temp = df[(df[col+"_use"] == 1) & (df["cluster1"] == core) & (df["core"] == True)]
                col_num = each_col
                if i < remain_num:
                    col_num += 1
                df_temp = df_temp.sample(col_num)
                df_total = pd.concat([df_total, df_temp], axis=0)
                i = i + 1
        else:
            col_num = core_num
            df_temp = df[(df["cluster1"] == core) & (df["core"] == True)]
            df_temp = df_temp.sample(col_num)

            df_total = pd.concat([df_total, df_temp], axis=0)
    df_total = df_total.sort_index()
    queries = []
    for index, row in df_total.iterrows():
        row_info = dict(row)
        queries.append(row_info["query_no"])
    df_total = df_total.sort_values(by="query_no")
    return queries
