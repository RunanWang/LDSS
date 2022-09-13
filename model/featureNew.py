import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.log import Log

L = Log(__name__).get_logger()


class WideDeepDataset(Dataset):
    def __init__(self, wide, deep, co, sql, labels):
        self.X_wide = wide
        self.X_deep = deep
        self.X_co = co
        self.X_sql = sql
        self.Y = labels

    def __getitem__(self, idx):
        xw = self.X_wide[idx]
        xd = self.X_deep[idx]
        xc = self.X_co[idx]
        xs = self.X_sql[idx]
        y = self.Y[idx]

        return xw, xd, xc, xs, y

    def __len__(self):
        return len(self.Y)


def load_data(data_path, name, log, device, table_columns, shffule=True):
    total_df = pd.read_pickle(data_path)
    origin_label_np = total_df["label"]
    temp_label_df = pd.DataFrame(origin_label_np)
    temp_label_df = temp_label_df.rename(columns={0: 'label'})
    total_label_df = temp_label_df.astype(np.float32)
    log.info(f"Data num={total_label_df.count()[0]} label1 num={total_label_df.mean()[0]*100:.2f}%")
    ################################################################################
    # 处理features
    data = total_df.drop(['label_est_card', 'label_GT_card', 'label_row_num',
                         'seed', 'id_query_no', 'id_sample_name', 'label', "label_std_qerror"], axis=1)

    def f(x): return 1 if x == 'TRUE' else 0
    data['replace'] = data['replace'].apply(f)
    # log.info(name + " data shape" + str(data.shape))
    data = data.astype(np.float32)
    # 处理wide部分
    wide_data = data[['replace', 'ratio', 'sample_type']]
    wide_data = pd.DataFrame()
    for col_name in table_columns:
        std_df = (data.loc[:, "sample_" + col_name + "_std"] * data.loc[:, col_name + "_use"]) ** 2
        std_df = np.sqrt(std_df)
        std_df.name = "sample_" + col_name + "_std"
        std_df = (std_df - std_df.min()) / (std_df.max() - std_df.min())
        mean_df = (data.loc[:, "sample_" + col_name + "_mean"] * data.loc[:, col_name + "_use"]) ** 2
        mean_df = np.sqrt(mean_df)
        mean_df.name = "sample_" + col_name + "_mean"
        mean_df = (mean_df - mean_df.min()) / (mean_df.max() - mean_df.min())
        try:
            sample_on_df = data.loc[:, "sample_on_" + col_name]
            sample_on_df.name = "sample_on_" + col_name
            wide_data = pd.concat([wide_data, std_df, mean_df, sample_on_df], axis=1)
        except:
            wide_data = pd.concat([wide_data, std_df, mean_df], axis=1)

    wide_data = wide_data.fillna(0)

    # 处理bin，对数据表的每一列进行处理，用直方图的值乘以query在这个区间上是否存在，得到每个区间上的差距
    bin_names = []
    finder = "query_"
    for col_name in data.columns:
        if col_name.find(finder) == 0:
            bin_names.append(col_name[len(finder):])
    bin_df_list = []
    for bin_name in bin_names:
        bin_df = (data.loc[:, "sample_" + bin_name] * data.loc[:, "query_" + bin_name]) ** 2
        bin_df = np.sqrt(bin_df)
        bin_df.name = bin_name
        bin_df_list.append(bin_df)
    bin_data = pd.concat(bin_df_list, axis=1)

    # 一维直方图特征加入wide
    sel_lists = []
    for col_name in table_columns:
        col_feats = []
        for feat_name in bin_data.columns:
            if col_name in feat_name:
                col_feats.append(bin_data.loc[:, feat_name])
        col_feat_df = pd.concat(col_feats, axis=1)
        col_sel_df = col_feat_df.sum(axis=1)
        col_sel_df.name = col_name + "_sel1"
        sel_lists.append(col_sel_df)
    col_sel_df = pd.concat(sel_lists, axis=1)
    col_sel_df = (col_sel_df - col_sel_df.min().min()) / (col_sel_df.max().max() - col_sel_df.min().min())
    wide_data = pd.concat([wide_data, col_sel_df], axis=1)

    # 其余的进行归一化，进入deep
    bin_df_list = []
    for col_name in table_columns:
        col_feats = []
        for feat_name in bin_data.columns:
            if col_name in feat_name:
                col_feats.append(bin_data.loc[:, feat_name])
        col_feat_df = pd.concat(col_feats, axis=1)
        col_feat_df = (col_feat_df - col_feat_df.min().min()) / (col_feat_df.max().max() - col_feat_df.min().min())

        bin_df_list.append(col_feat_df)
    deep_data = pd.concat(bin_df_list, axis=1)

    # SQL特征单独编码
    sql_feats = []
    for col_name in table_columns:
        sql_feats.append(data.loc[:, col_name + "_ql"])
        sql_feats.append(data.loc[:, col_name + "_qr"])
        data = data.drop([col_name + "_ql", col_name + "_qr"], axis=1)
    sql_data = pd.concat(sql_feats, axis=1)

    # 多维特征
    co_feats = []
    col_name_done = []
    for col_name1 in table_columns:
        for col_name2 in table_columns:
            if col_name1 + col_name2 in col_name_done:
                pass
            elif col_name2 + col_name1 in col_name_done:
                pass
            elif col_name1 == col_name2:
                pass
            else:
                try:
                    co_df = (data.loc[:, "sample_co_" + col_name1 + "_" + col_name2] *
                             data.loc[:, col_name1 + "_use"] * data.loc[:, col_name2 + "_use"]) ** 2
                except KeyError:
                    co_df = (data.loc[:, "sample_co_" + col_name2 + "_" + col_name1] *
                             data.loc[:, col_name1 + "_use"] * data.loc[:, col_name2 + "_use"]) ** 2
                col_name_done.append(col_name1 + col_name2)
                co_feats.append(co_df)
                data = data.drop(["sample_co_" + col_name1 + "_" + col_name2], axis=1)
    co_data = pd.concat(co_feats, axis=1)
    co_data = (co_data - co_data.min().min()) / (co_data.max().max() - co_data.min().min())

    # 合并
    deep_data = pd.concat([wide_data, deep_data], axis=1)
    wide_data = wide_data.astype(np.float32)
    deep_data = deep_data.astype(np.float32)
    co_data = co_data.astype(np.float32)
    class1_num = total_label_df.count()[0] * total_label_df.mean()[0]
    class0_num = total_label_df.count()[0] - class1_num
    data_shape = {"wide": wide_data.shape[1], "deep": deep_data.shape[1], "co": co_data.shape[1],
                  "sql": sql_data.shape[1], "class0": class0_num, "class1": class1_num}
    train_dataset = WideDeepDataset(torch.tensor(wide_data.values).to(device),
                                    torch.tensor(deep_data.values).to(device),
                                    torch.tensor(co_data.values).to(device),
                                    torch.tensor(sql_data.values).to(device),
                                    torch.tensor(total_label_df.values).to(device))
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=shffule, num_workers=0)
    return total_label_df, loader, total_label_df.shape[0], data_shape


def load_data_gbdt(data_path, name, log, table_columns):
    total_df = pd.read_pickle(data_path)
    origin_label_np = total_df["label"]
    temp_label_df = pd.DataFrame(origin_label_np)
    temp_label_df = temp_label_df.rename(columns={0: 'label'})
    total_label_df = temp_label_df.astype(np.float32)
    log.info(f"Label Count:{total_label_df.count()[0]} Mean:{total_label_df.mean()[0]}")
    ################################################################################
    # 处理features
    data = total_df.drop(['label_est_card', 'label_GT_card', 'label_row_num',
                         'seed', 'id_query_no', 'id_sample_name', 'label', "label_std_qerror"], axis=1)

    def f(x): return 1 if x == 'TRUE' else 0
    data['replace'] = data['replace'].apply(f)
    log.info(name + " data shape" + str(data.shape))
    data = data.astype(np.float32)
    # 处理wide部分
    wide_data = data[['replace', 'ratio', 'sample_type']]
    wide_data = pd.DataFrame()
    for col_name in table_columns:
        std_df = data.loc[:, "sample_" + col_name + "_std"] * data.loc[:, col_name + "_use"]
        std_df.name = "sample_" + col_name + "_std"
        mean_df = data.loc[:, "sample_" + col_name + "_mean"] * data.loc[:, col_name + "_use"]
        mean_df.name = "sample_" + col_name + "_mean"
        try:
            sample_on_df = data.loc[:, "sample_on_" + col_name]
            sample_on_df.name = "sample_on_" + col_name
            wide_data = pd.concat([wide_data, std_df, mean_df, sample_on_df], axis=1)
        except:
            wide_data = pd.concat([wide_data, std_df, mean_df], axis=1)

    wide_data = wide_data.fillna(0)

    # 处理bin，对数据表的每一列进行处理，用直方图的值乘以query在这个区间上是否存在，得到每个区间上的差距
    bin_names = []
    finder = "query_"
    for col_name in data.columns:
        if col_name.find(finder) == 0:
            bin_names.append(col_name[len(finder):])
    bin_df_list = []
    for bin_name in bin_names:
        bin_df = data.loc[:, "sample_" + bin_name] * data.loc[:, "query_" + bin_name]
        bin_df.name = bin_name
        bin_df_list.append(bin_df)
    bin_data = pd.concat(bin_df_list, axis=1)

    # 一维直方图特征加入wide
    sel_lists = []
    for col_name in table_columns:
        col_feats = []
        for feat_name in bin_data.columns:
            if col_name in feat_name:
                col_feats.append(bin_data.loc[:, feat_name])
        col_feat_df = pd.concat(col_feats, axis=1)
        col_sel_df = col_feat_df.sum(axis=1)
        col_sel_df.name = col_name + "_sel1"
        sel_lists.append(col_sel_df)
    col_sel_df = pd.concat(sel_lists, axis=1)
    wide_data = pd.concat([wide_data, col_sel_df], axis=1)

    # 其余的进行归一化，进入deep
    bin_df_list = []
    feat_name_list = []
    for feat_name in bin_data.columns:
        if col_name in feat_name:
            if feat_name in feat_name_list:
                pass
            else:
                bin_df_list.append(bin_data.loc[:, feat_name])
                feat_name_list.append(feat_name)
    deep_data = pd.concat(bin_df_list, axis=1)

    # SQL特征单独编码
    sql_feats = []
    for col_name in table_columns:
        sql_feats.append(data.loc[:, col_name + "_ql"])
        sql_feats.append(data.loc[:, col_name + "_qr"])
        data = data.drop([col_name + "_ql", col_name + "_qr"], axis=1)
    sql_data = pd.concat(sql_feats, axis=1)

    # 多维特征
    co_feats = []
    col_name_done = []
    for col_name1 in table_columns:
        for col_name2 in table_columns:
            if col_name1 + col_name2 in col_name_done:
                pass
            elif col_name2 + col_name1 in col_name_done:
                pass
            elif col_name1 == col_name2:
                pass
            else:
                try:
                    co_df = data.loc[:, "sample_co_" + col_name1 + "_" + col_name2] * \
                        data.loc[:, col_name1 + "_use"] * data.loc[:, col_name2 + "_use"]
                except KeyError:
                    co_df = data.loc[:, "sample_co_" + col_name2 + "_" + col_name1] * \
                        data.loc[:, col_name1 + "_use"] * data.loc[:, col_name2 + "_use"]
                col_name_done.append(col_name1 + col_name2)
                co_feats.append(co_df)
                data = data.drop(["sample_co_" + col_name1 + "_" + col_name2], axis=1)
    co_data = pd.concat(co_feats, axis=1)

    # 合并
    wide_data = wide_data.astype(np.float32)
    deep_data = deep_data.astype(np.float32)
    co_data = co_data.astype(np.float32)
    X = pd.concat([wide_data, deep_data, co_data, sql_data], axis=1)
    y = total_label_df.values.ravel()
    return X, y


def load_data_multi(data_path, name, log, device, table_col_dist, join_col, shffule=True):
    log.info(f"Start loading pickle of {name}...")
    total_df = pd.read_pickle(data_path)
    log.info(f"Pickle of {name} loaded, filling nan...")
    total_df = total_df.fillna(0)
    origin_label_np = total_df["label"]
    temp_label_df = pd.DataFrame(origin_label_np)
    temp_label_df = temp_label_df.rename(columns={0: 'label'})
    total_label_df = temp_label_df.astype(np.float32)
    log.info(f"Data num={total_label_df.count()[0]} label1 num={total_label_df.mean()[0]*100:.2f}%")
    ################################################################################
    # 处理features
    discard_cols = []
    wide_feats = []
    for col_name in total_df.columns:
        col_type = col_name.split("_")[0]
        if col_type == "label" or col_type == "id":
            discard_cols.append(col_name)
        elif col_type == "of":
            wide_feats.append(col_name)
    data = total_df.drop(discard_cols, axis=1)
    log.info(f"Dropping label and id, {name} raw data shape: {str(data.shape)}")
    data = data.astype(np.float32)
    # 处理uc和ut
    for table_name, col_name in join_col.items():
        data[f"uc_{table_name}_{col_name}"] = data[f"uc_{table_name}_{col_name}"] + data[f"ut_{table_name}_{col_name}"]\
            # 处理wide部分
    log.info(f"{name} generating wide feats...")
    # of类特征
    wide_data = data[wide_feats]
    for table_name, col_names in table_col_dist.items():
        for col_name in col_names:
            std_df = (data.loc[:, f"sf_std_{table_name}_{col_name}"] * data.loc[:, f"uc_{table_name}_{col_name}"]) ** 2
            std_df = np.sqrt(std_df)
            std_df.name = f"sf_std_{table_name}_{col_name}"
            mean_df = (data.loc[:, f"sf_mean_{table_name}_{col_name}"]
                       * data.loc[:, f"uc_{table_name}_{col_name}"]) ** 2
            mean_df = np.sqrt(mean_df)
            mean_df.name = f"sf_mean_{table_name}_{col_name}"
            wide_data = pd.concat([wide_data, std_df, mean_df], axis=1)
    wide_data = (wide_data - wide_data.min()) / (wide_data.max() - wide_data.min())
    log.info(f"{name} of data shape: {str(wide_data.shape)}")
    # 处理bin，对数据表的每一列进行处理，用直方图的值乘以query在这个区间上是否存在，得到每个区间上的差距
    bin_names = []
    finder = "qh_"
    for col_name in data.columns:
        if col_name.find(finder) == 0:
            bin_names.append(col_name[len(finder):])
    # bin_value * query_value
    bin_df_list = []
    for bin_name in bin_names:
        bin_df = (data.loc[:, "hf_" + bin_name] * data.loc[:, "qh_" + bin_name]) ** 2
        bin_df = np.sqrt(bin_df)
        bin_df.name = bin_name
        bin_df_list.append(bin_df)
    bin_data = pd.concat(bin_df_list, axis=1)
    # 一维直方图特征加入wide
    sel_lists = []
    for table_name, c_names in table_col_dist.items():
        for c_name in c_names:
            col_name = f"{table_name}_{c_name}"
            col_feats = []
            for feat_name in bin_data.columns:
                if col_name in feat_name:
                    col_feats.append(bin_data.loc[:, feat_name])
            col_feat_df = pd.concat(col_feats, axis=1)
            col_sel_df = col_feat_df.sum(axis=1)
            col_sel_df.name = col_name + "_sel"
            sel_lists.append(col_sel_df)
    col_sel_df = pd.concat(sel_lists, axis=1)
    col_sel_df = (col_sel_df - col_sel_df.min().min()) / (col_sel_df.max().max() - col_sel_df.min().min())
    wide_data = pd.concat([wide_data, col_sel_df], axis=1)
    wide_data = wide_data.fillna(0)
    log.info(f"{name} col_sel data shape: {str(col_sel_df.shape)}")

    # 其余的进行归一化，进入deep
    bin_df_list = []
    for table_name, c_names in table_col_dist.items():
        for c_name in c_names:
            col_name = f"{table_name}_{c_name}"
            col_feats = []
            for feat_name in bin_data.columns:
                if col_name in feat_name:
                    col_feats.append(bin_data.loc[:, feat_name])
            col_feat_df = pd.concat(col_feats, axis=1)
            col_feat_df = (col_feat_df - col_feat_df.min().min()) / (col_feat_df.max().max() - col_feat_df.min().min())
            bin_df_list.append(col_feat_df)
    deep_data = pd.concat(bin_df_list, axis=1)
    deep_data = deep_data.fillna(0)
    log.info(f"{name} deep data shape: {str(deep_data.shape)}")

    # SQL特征单独编码
    sql_feats = []
    for table_name, c_names in table_col_dist.items():
        for c_name in c_names:
            col_name = f"qf_{table_name}_{c_name}"
            sql_feats.append(data.loc[:, col_name+"_ql"])
            sql_feats.append(data.loc[:, col_name+"_qr"])
            data = data.drop([col_name+"_ql", col_name+"_qr"], axis=1)
    sql_data = pd.concat(sql_feats, axis=1)
    sql_data = sql_data.fillna(0)
    log.info(f"{name} sql data shape: {str(sql_data.shape)}")

    # 多维特征
    co_feats = []
    for table_name, table_columns in table_col_dist.items():
        col_name_done = []
        for col_name1 in table_columns:
            for col_name2 in table_columns:
                if col_name1 + col_name2 in col_name_done:
                    pass
                elif col_name2 + col_name1 in col_name_done:
                    pass
                elif col_name1 == col_name2:
                    pass
                else:
                    try:
                        co_df = (data.loc[:, f"sf_co_{table_name}_{col_name1}_2_{col_name2}"] * data.loc[:,
                                 f"uc_{table_name}_{col_name1}"] * data.loc[:, f"uc_{table_name}_{col_name2}"]) ** 2
                        data = data.drop([f"sf_co_{table_name}_{col_name1}_2_{col_name2}"], axis=1)
                    except KeyError:
                        co_df = (data.loc[:, f"sf_co_{table_name}_{col_name2}_2_{col_name1}"] * data.loc[:,
                                 f"uc_{table_name}_{col_name1}"] * data.loc[:, f"uc_{table_name}_{col_name2}"]) ** 2
                        data = data.drop([f"sf_co_{table_name}_{col_name2}_2_{col_name1}"], axis=1)
                    col_name_done.append(col_name1 + col_name2)
                    co_feats.append(co_df)

    co_data = pd.concat(co_feats, axis=1)
    co_data = (co_data - co_data.min().min()) / (co_data.max().max() - co_data.min().min())
    co_data = co_data.fillna(0)
    log.info(f"{name} co data shape: {str(co_data.shape)}")

    # 合并
    class1_num = total_label_df.count()[0] * total_label_df.mean()[0]
    class0_num = total_label_df.count()[0] - class1_num
    wide_data = wide_data.astype(np.float32)
    deep_data = deep_data.astype(np.float32)
    sql_data = sql_data.astype(np.float32)
    co_data = co_data.astype(np.float32)
    data_shape = {"wide": wide_data.shape[1], "deep": deep_data.shape[1],
                  "sql": sql_data.shape[1], "co": co_data.shape[1], "class0": class0_num, "class1": class1_num}
    train_dataset = WideDeepDataset(torch.tensor(wide_data.values).to(device),
                                    torch.tensor(deep_data.values).to(device),
                                    torch.tensor(co_data.values).to(device),
                                    torch.tensor(sql_data.values).to(device),
                                    torch.tensor(total_label_df.values).to(device))
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=shffule, num_workers=0)
    return total_label_df, loader, total_label_df.shape[0], data_shape


def load_data_multi_gbdt(data_path, name, log, table_col_dist, join_col):
    log.info(f"Start loading pickle of {name}...")
    total_df = pd.read_pickle(data_path)
    log.info(f"Pickle of {name} loaded, filling nan...")
    total_df = total_df.fillna(0)
    origin_label_np = total_df["label"]
    temp_label_df = pd.DataFrame(origin_label_np)
    temp_label_df = temp_label_df.rename(columns={0: 'label'})
    total_label_df = temp_label_df.astype(np.float32)
    log.info(f"Data num={total_label_df.count()[0]} label1 num={total_label_df.mean()[0]*100:.2f}%")
    ################################################################################
    # 处理features
    discard_cols = []
    wide_feats = []
    for col_name in total_df.columns:
        col_type = col_name.split("_")[0]
        if col_type == "label" or col_type == "id":
            discard_cols.append(col_name)
        elif col_type == "of":
            wide_feats.append(col_name)
    data = total_df.drop(discard_cols, axis=1)
    log.info(f"Dropping label and id, {name} raw data shape: {str(data.shape)}")
    data = data.astype(np.float32)
    # 处理uc和ut
    for table_name, col_name in join_col.items():
        data[f"uc_{table_name}_{col_name}"] = data[f"uc_{table_name}_{col_name}"] + data[f"ut_{table_name}_{col_name}"]\
            # 处理wide部分
    log.info(f"{name} generating wide feats...")
    # of类特征
    wide_data = data[wide_feats]
    for table_name, col_names in table_col_dist.items():
        for col_name in col_names:
            std_df = (data.loc[:, f"sf_std_{table_name}_{col_name}"] * data.loc[:, f"uc_{table_name}_{col_name}"]) ** 2
            std_df = np.sqrt(std_df)
            std_df.name = f"sf_std_{table_name}_{col_name}"
            mean_df = (data.loc[:, f"sf_mean_{table_name}_{col_name}"]
                       * data.loc[:, f"uc_{table_name}_{col_name}"]) ** 2
            mean_df = np.sqrt(mean_df)
            mean_df.name = f"sf_mean_{table_name}_{col_name}"
            wide_data = pd.concat([wide_data, std_df, mean_df], axis=1)
    wide_data = (wide_data - wide_data.min()) / (wide_data.max() - wide_data.min())
    log.info(f"{name} of data shape: {str(wide_data.shape)}")
    # 处理bin，对数据表的每一列进行处理，用直方图的值乘以query在这个区间上是否存在，得到每个区间上的差距
    bin_names = []
    finder = "qh_"
    for col_name in data.columns:
        if col_name.find(finder) == 0:
            bin_names.append(col_name[len(finder):])
    # bin_value * query_value
    bin_df_list = []
    for bin_name in bin_names:
        bin_df = (data.loc[:, "hf_" + bin_name] * data.loc[:, "qh_" + bin_name]) ** 2
        bin_df = np.sqrt(bin_df)
        bin_df.name = bin_name
        bin_df_list.append(bin_df)
    bin_data = pd.concat(bin_df_list, axis=1)
    # 一维直方图特征加入wide
    sel_lists = []
    for table_name, c_names in table_col_dist.items():
        for c_name in c_names:
            col_name = f"{table_name}_{c_name}"
            col_feats = []
            for feat_name in bin_data.columns:
                if col_name in feat_name:
                    col_feats.append(bin_data.loc[:, feat_name])
            col_feat_df = pd.concat(col_feats, axis=1)
            col_sel_df = col_feat_df.sum(axis=1)
            col_sel_df.name = col_name + "_sel"
            sel_lists.append(col_sel_df)
    col_sel_df = pd.concat(sel_lists, axis=1)
    col_sel_df = (col_sel_df - col_sel_df.min().min()) / (col_sel_df.max().max() - col_sel_df.min().min())
    wide_data = pd.concat([wide_data, col_sel_df], axis=1)
    wide_data = wide_data.fillna(0)
    log.info(f"{name} col_sel data shape: {str(col_sel_df.shape)}")

    # 其余的进行归一化，进入deep
    bin_df_list = []
    feat_name_list = []
    for table_name, c_names in table_col_dist.items():
        for c_name in c_names:
            col_name = f"{table_name}_{c_name}"
            col_feats = []
            for feat_name in bin_data.columns:
                if col_name in feat_name:
                    if feat_name in feat_name_list:
                        pass
                    else:
                        col_feats.append(bin_data.loc[:, feat_name])
                        feat_name_list.append(feat_name)
    deep_data = pd.concat(col_feats, axis=1)
    deep_data = deep_data.fillna(0)

    # SQL特征单独编码
    sql_feats = []
    for table_name, c_names in table_col_dist.items():
        for c_name in c_names:
            col_name = f"qf_{table_name}_{c_name}"
            sql_feats.append(data.loc[:, col_name+"_ql"])
            sql_feats.append(data.loc[:, col_name+"_qr"])
            data = data.drop([col_name+"_ql", col_name+"_qr"], axis=1)
    sql_data = pd.concat(sql_feats, axis=1)
    sql_data = sql_data.fillna(0)
    log.info(f"{name} sql data shape: {str(sql_data.shape)}")

    # 多维特征
    co_feats = []
    for table_name, table_columns in table_col_dist.items():
        col_name_done = []
        for col_name1 in table_columns:
            for col_name2 in table_columns:
                if col_name1 + col_name2 in col_name_done:
                    pass
                elif col_name2 + col_name1 in col_name_done:
                    pass
                elif col_name1 == col_name2:
                    pass
                else:
                    try:
                        co_df = (data.loc[:, f"sf_co_{table_name}_{col_name1}_2_{col_name2}"] * data.loc[:,
                                 f"uc_{table_name}_{col_name1}"] * data.loc[:, f"uc_{table_name}_{col_name2}"]) ** 2
                        data = data.drop([f"sf_co_{table_name}_{col_name1}_2_{col_name2}"], axis=1)
                    except KeyError:
                        co_df = (data.loc[:, f"sf_co_{table_name}_{col_name2}_2_{col_name1}"] * data.loc[:,
                                 f"uc_{table_name}_{col_name1}"] * data.loc[:, f"uc_{table_name}_{col_name2}"]) ** 2
                        data = data.drop([f"sf_co_{table_name}_{col_name2}_2_{col_name1}"], axis=1)
                    col_name_done.append(col_name1 + col_name2)
                    co_feats.append(co_df)

    co_data = pd.concat(co_feats, axis=1)
    co_data = (co_data - co_data.min().min()) / (co_data.max().max() - co_data.min().min())
    co_data = co_data.fillna(0)
    log.info(f"{name} co data shape: {str(co_data.shape)}")

    # 合并
    wide_data = wide_data.astype(np.float32)
    deep_data = deep_data.astype(np.float32)
    sql_data = sql_data.astype(np.float32)
    co_data = co_data.astype(np.float32)
    X = pd.concat([wide_data, deep_data, co_data, sql_data], axis=1)
    y = total_label_df.values.ravel()
    return X, y
