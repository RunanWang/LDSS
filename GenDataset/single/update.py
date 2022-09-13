import pandas as pd
import numpy as np
import pickle
import torch
from multiprocessing import Process, Queue

from utils.log import Log, data_describe_log
from utils.metrics import cal_q_error

from GenDataset.single.dataset.dataset import Table
from GenDataset.single.dataset.feat import Histogram
from GenDataset.single.workload.gen_label import generate_label_for_query, generate_label_by_standard_sample
from GenDataset.single.workload.workload import load_gen_queryset, query_2_vec, query_2_sql, query_2_histogram_vec
from GenDataset.single.estimator.sample import Sampling
from GenDataset.single.estimator.sample_group import SamplingGroupUpdate
from GenDataset.single.workload.cluster import get_small_queryset
from model.featureNew import load_data
import constants as C
import model.recommend as R
from model.model import WideDeepNew
from model.train import BCELoss_class_weighted, train_one_epoch_new
from model.test import get_rec_info, get_score_new, cal_three_ratio
from utils.metrics import confusion

L = Log(__name__, "update").get_logger()


def csv_to_pkl(table_name: str):
    table_path = C.DATA_ROOT / "dmv" / "update" / "table" / f"{table_name}.csv"
    temp_table_path = C.TEMP_ROOT / "table" / "update"
    if not temp_table_path.exists():
        temp_table_path.mkdir()
    pkl_path = temp_table_path / f"{table_name}.pkl"
    if pkl_path.is_file():
        return
    df = pd.read_csv(table_path)
    df.to_pickle(pkl_path)
    L.info("csv to pkl in path " + str(pkl_path))


def load_table(table_name: str) -> Table:
    update_path = C.TEMP_ROOT / "update"
    if not update_path.exists():
        update_path.mkdir()
    obj_path = update_path / "obj"
    if not obj_path.exists():
        obj_path.mkdir()
    table_path = obj_path / f"{table_name}.table.pkl"
    if table_path.is_file():
        L.info("table exists, load...")
        with open(table_path, 'rb') as f:
            table = pickle.load(f)
        L.info(f"load finished: {table}")
        return table
    else:
        origin_path = C.TEMP_ROOT / "table" / "update" / f"{table_name}.pkl"
        csv_to_pkl(table_name)
        data = pd.read_pickle(origin_path)
        table = Table(data, table_name)
        L.info("write table to disk...")
        with open(table_path, 'wb') as f:
            pickle.dump(table, f, protocol=C.PKL_PROTO)
        return table


def load_histo(table_name: str, df_total: pd.DataFrame, overwrite: bool = False) -> Histogram:
    obj_path = C.TEMP_ROOT / "update" / "obj"
    if not obj_path.exists():
        obj_path.mkdir()
    table_path = obj_path / f"{table_name}.histo.pkl"
    if not overwrite and table_path.is_file():
        L.info("Histo exists, load...")
        with open(table_path, 'rb') as f:
            data = pickle.load(f)
        L.info(f"load finished: {table_name}")
        return data
    else:
        L.info("Histo exists, load...")
        data = Histogram(df_total)
        L.info("write data to disk...")
        with open(table_path, 'wb') as f:
            pickle.dump(data, f, protocol=C.PKL_PROTO)
        return data


def load_sample_group(table, dataset_name):
    sample_group_path = C.TEMP_ROOT / "update"
    C.mkdir(sample_group_path)
    sample_group_path = sample_group_path / f"{dataset_name}-sg.pkl"
    if sample_group_path.is_file():
        L.info("sample group exists, load...")
        with open(sample_group_path, 'rb') as f:
            sample_group = pickle.load(f)
        L.info("load finished.")
        return sample_group
    else:
        sample_group = SamplingGroupUpdate(table)
        with open(sample_group_path, 'wb') as f:
            pickle.dump(sample_group, f, protocol=C.PKL_PROTO)
        return sample_group


def load_sample_feature(sg: SamplingGroupUpdate, dataset_name: str, total_df, rel_df):
    sample_group_path = C.TEMP_ROOT / "update"
    C.mkdir(sample_group_path)
    sample_feature_path = sample_group_path / f"{dataset_name}-sf.pkl"
    if sample_feature_path.is_file():
        L.info("sample features exists, load...")
        with open(sample_feature_path, 'rb') as f:
            sample_feat = pickle.load(f)
        L.info("load finished.")
        return sample_feat
    else:
        sample_feat = sg.feature(total_df, rel_df)
        with open(sample_feature_path, 'wb') as f:
            pickle.dump(sample_feat, f, protocol=C.PKL_PROTO)
        return sample_feat


def cal_query_set(id, queries, query_num_offset, table, base_table, histo, standard_sample, sample_group, sample_feature, q: Queue):
    rec_for_queryset = []
    query_info_list = []
    for i, query in enumerate(queries):
        rec_for_one_query, query_info = cal_vec(
            table, base_table, query, query_num_offset + i, histo, standard_sample, sample_group, sample_feature)
        rec_for_queryset += rec_for_one_query
        query_info_list.append(query_info)
        if (i + 1) % 1000 == 0:
            L.info(f"Query of group {id:>2d} - {(i + 1):>6d} / {len(queries)}")
    q.put((rec_for_queryset, query_info_list))


def cal_vec(table, base_table, query, query_num, histo, standard_sample, sample_group, sample_feature):
    # calculate true cardinality
    rec_for_one_query = []
    gt_card = generate_label_for_query(table, query)

    # cardinality estimation by sample group and standard sample
    std_card, std_cost = standard_sample.query2(query)
    standard_q_error = cal_q_error(gt_card, std_card)
    card, cost = sample_group.query2(query)

    # gnenrate features for query
    histo_feat = query_2_histogram_vec(query, histo)
    query_vec = query_2_vec(query, base_table)
    query_sql = query_2_sql(query, base_table, aggregate=False, dbms='postgres')
    query_info = {"query_no": query_num, "query": query_sql, "cardinality_true": gt_card, 'est_std': std_card}
    query_info.update(query_vec)
    query_vec.update(histo_feat)
    query_vec['id_query_no'] = query_num
    query_vec['label_GT_card'] = gt_card
    query_vec['label_std_qerror'] = standard_q_error
    query_vec['label_row_num'] = table.row_num
    query_vec['label_std_time'] = std_cost

    # packing query and sample together
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
        query_info['est_'+sample_name] = sample_card
        rec_for_one_query.append(record)
    return rec_for_one_query, query_info


def gen_dataset(queryset, table, base_table, histo, standard_sample, sampling_group, sample_features, dataset_name, need_query_info):
    mp_num = 64
    q = Queue()
    for group, queries in queryset.items():
        result_dir = C.DATASET_PATH / "update"
        C.mkdir(result_dir)
        pkl_path = result_dir / f"{dataset_name}-{group}.pkl"
        if pkl_path.is_file():
            L.info(f"Dataset: {dataset_name} - {group} exists, load")
            continue

        if group == "train":
            queries = queries[:10000]
        else:
            queries = queries[:1000]
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
                        args=(i, queries_in_mp, start_id, table, base_table, histo, standard_sample, sampling_group, sample_features, q,))
            p.start()
            p_list.append(p)
        L.info("All batch ditributed!")
        total_rec_list = []
        total_query_info_list = []
        for p in p_list:
            rec_for_queryset, query_info_list = q.get()
            total_rec_list += rec_for_queryset
            total_query_info_list += query_info_list
        for p in p_list:
            p.join()

        L.info("Saving records of group " + group)
        df = pd.DataFrame(total_rec_list)
        df.to_pickle(pkl_path)
        if need_query_info and group == "train":
            csv_path = result_dir / f"q-{dataset_name}.csv"
            df = pd.DataFrame(total_query_info_list)
            df.to_csv(csv_path)


def cal_append(append_name, queryset, table_75, table_100, histo):
    standard_sample_75 = Sampling(table_75, C.STANDARD_SAMPLE_PAR["ratio"], C.STANDARD_SAMPLE_PAR["seed"])
    sampling_group_75 = load_sample_group(table_75, "dmv-75")
    sample_features_75 = load_sample_feature(sampling_group_75, "dmv-75", table_75.data, standard_sample_75.sample)
    table_append = load_table(append_name)
    append_data_df = pd.read_csv(C.DATA_ROOT / "dmv" / "update" / "append" / f"{append_name}.csv")
    sampling_group_append = sampling_group_75.append_data(append_data_df)
    standard_sample_append = standard_sample_75.append_data(append_data_df)

    # only append sample
    queryset_1 = {"test": queryset["test"]}
    gen_dataset(queryset_1, table_append, table_100, histo, standard_sample_append, sampling_group_append, sample_features_75, f"{append_name}-sample-only", False)
    # fine tuning
    queryset_2 = {"tuning": queryset["tuning"]}
    gen_dataset(queryset_2, table_append, table_100, histo, standard_sample_append, sampling_group_append, sample_features_75, f"{append_name}", False)
    # retraining
    queryset_3 = {"train": queryset["train"], "valid": queryset["valid"], "test": queryset["test"]}
    sample_features_append = load_sample_feature(sampling_group_append, f"dmv-{append_name}", table_100.data, standard_sample_append.sample)
    gen_dataset(queryset_3, table_append, table_100, histo, standard_sample_append, sampling_group_append, sample_features_append, f"{append_name}", False)


def train_model(dataset_name, train_data_path, valid_data_path, table_col):
    # 1. Load Data
    model_path = C.MODEL_PATH / f"{dataset_name}.pkl"
    if model_path.is_file():
        L.info(f"{dataset_name} model exist")
        return
    train_label_df, train_loader, train_size, input_size = load_data(train_data_path, "train", L, C.DEVICE, table_col)
    valid_label_df, valid_loader, _, _ = load_data(valid_data_path, "valid", L, C.DEVICE, table_col)
    L.info("Model input: " + str(input_size))

    # 2. Setup Model
    torch.manual_seed(521)
    model = WideDeepNew(input_size)
    L.info("Model established: " + str(model))
    model.to(C.DEVICE)
    class0_weight = max(input_size["class0"], input_size["class1"]) / input_size["class0"]
    class1_weight = max(input_size["class0"], input_size["class1"]) / input_size["class1"]
    weights = [class0_weight, class1_weight]
    class_weights = torch.FloatTensor(weights)
    L.info(f"Loss Weights: {class_weights}")
    loss_function = BCELoss_class_weighted(class_weights)
    L.info("Using BCELoss as loss function.")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    L.info("Using Adam as optimization function.")

    # 3. Training Model
    L.info("Begin the training of model.")
    for epoch in range(C.EPOCHES):
        train_losses, train_acc, valid_losses, valid_acc = train_one_epoch_new(
            model, loss_function, train_loader, valid_loader, optimizer)
        L.info('epoch : {}, train loss : {:.4f}, train acc: {:.4f}, valid loss : {:.4f}, valid acc : {:.4f}'.format(
            epoch + 1, np.mean(train_losses), np.mean(train_acc), np.mean(valid_losses), np.mean(valid_acc)))
    with torch.no_grad():
        ttp, tfp, ttn, tfn = 0, 0, 0, 0
        for i, (wide_feat, deep_feat, co_feat, sql_feat, label) in enumerate(valid_loader):
            output = model(wide_feat, deep_feat, co_feat, sql_feat)
            # label_arg_max = torch.argmax(label, dim=1)
            tp, fp, tn, fn = confusion(output, label)
            ttp += tp
            tfp += fp
            ttn += tn
            tfn += fn
    total = ttp + tfp + ttn + tfn
    metrics = {"tp": round(ttp/total, 3), "fp": round(tfp/total, 3),
               "tn": round(ttn/total, 3), "fn": round(tfn/total, 3)}
    L.info(f"{metrics}")
    # 4. Saving Model

    L.info("Saving Model.")
    torch.save(model, model_path)
    L.info("Model saved in path " + str(model_path))


def tuning_model(old_model_path, dataset_name, train_data_path, valid_data_path, table_col):
    tuning_epoch = 20
    torch.manual_seed(521)
    # 1. Load Data
    model_path_new = C.MODEL_PATH / f"{dataset_name}-tuning.pkl"
    model = torch.load(old_model_path)
    train_label_df, train_loader, train_size, input_size = load_data(train_data_path, "train", L, C.DEVICE, table_col)
    valid_label_df, valid_loader, _, _ = load_data(valid_data_path, "valid", L, C.DEVICE, table_col)
    L.info("Model input: " + str(input_size))

    # 2. Setup Model
    model.to(C.DEVICE)
    class0_weight = max(input_size["class0"], input_size["class1"]) / input_size["class0"]
    class1_weight = max(input_size["class0"], input_size["class1"]) / input_size["class1"]
    weights = [class0_weight, class1_weight]
    class_weights = torch.FloatTensor(weights)
    L.info(f"Loss Weights: {class_weights}")
    loss_function = BCELoss_class_weighted(class_weights)
    L.info("Using BCELoss as loss function.")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    L.info("Using Adam as optimization function.")

    # 3. Training Model
    L.info("Begin the training of model.")
    for epoch in range(tuning_epoch):
        train_losses, train_acc, valid_losses, valid_acc = train_one_epoch_new(
            model, loss_function, train_loader, valid_loader, optimizer)
        L.info('epoch : {}, train loss : {:.4f}, train acc: {:.4f}, valid loss : {:.4f}, valid acc : {:.4f}'.format(
            epoch + 1, np.mean(train_losses), np.mean(train_acc), np.mean(valid_losses), np.mean(valid_acc)))
    with torch.no_grad():
        ttp, tfp, ttn, tfn = 0, 0, 0, 0
        for i, (wide_feat, deep_feat, co_feat, sql_feat, label) in enumerate(valid_loader):
            output = model(wide_feat, deep_feat, co_feat, sql_feat)
            # label_arg_max = torch.argmax(label, dim=1)
            tp, fp, tn, fn = confusion(output, label)
            ttp += tp
            tfp += fp
            ttn += tn
            tfn += fn
    total = ttp + tfp + ttn + tfn
    metrics = {"tp": round(ttp/total, 3), "fp": round(tfp/total, 3),
               "tn": round(ttn/total, 3), "fn": round(tfn/total, 3)}
    L.info(metrics)
    # 4. Saving Model

    L.info("Saving Model.")
    torch.save(model, model_path_new)
    L.info("Tuning Model saved in path " + str(model_path_new))


def get_test_result(model_path, test_path, workload_name, table_col):
    L.info("Loading model.")
    model = torch.load(model_path)
    L.info("Loading Data.")
    test_df = pd.read_pickle(test_path)
    test_df = test_df.fillna(0)
    _, tloader, _, _ = load_data(test_path, workload_name, L, C.DEVICE, table_col, False)

    model.eval()
    ttp, tfp, ttn, tfn = 0, 0, 0, 0
    with torch.no_grad():
        for _, (wide_feat, deep_feat, co_feat, sql_feat, label) in enumerate(tloader):
            output = model(wide_feat, deep_feat, co_feat, sql_feat)
            tp, fp, tn, fn = confusion(output, label)
            ttp += tp
            tfp += fp
            ttn += tn
            tfn += fn
    total = ttp + tfp + ttn + tfn
    metrics = {"tp": round(ttp/total, 3), "fp": round(tfp/total, 3),
               "tn": round(ttn/total, 3), "fn": round(tfn/total, 3)}
    L.info(f"On Test: {metrics}")

    L.info("Calculating Score from model.")
    test_df = test_df[["ratio", "id_query_no", "id_sample_name", "label_GT_card", "label_est_card",
                       "label_std_qerror", "label_q_error", "label_cost_time", "label_std_time", "label"]]
    _, tloader, _, _ = load_data(test_path, workload_name, L, C.DEVICE, table_col, False)
    query_sample_to_info = get_score_new(model, tloader, test_df)

    # Recommend
    L.info("Recommending sample from Meta-info.")
    ratio_ldss, rec_time_ldss, sample_time_ldss, std_time, error_ldss, error_std_list = get_rec_info(query_sample_to_info, R.smallest_sample_wins, L)

    # Evaluate
    L.info("Error of CE using LDSS")
    error_df = pd.DataFrame(error_ldss)
    data_describe_log(workload_name, error_df, L)
    L.info(f"Avg Sample Ratio={ratio_ldss:.4f}")

    cal_three_ratio(query_sample_to_info, R.smallest_sample_wins, L)


def update_workflow(dataset_name):
    # Load workload
    queryset = load_gen_queryset(dataset_name)

    # Load table
    table_75 = load_table("75")
    table_100 = load_table("100")
    histo = load_histo(dataset_name, table_100.data)

    # Load samples
    std_sample_ratio = C.STANDARD_SAMPLE_PAR["ratio"]
    standard_sample_75 = Sampling(table_75, std_sample_ratio, C.STANDARD_SAMPLE_PAR["seed"])
    sampling_group_75 = load_sample_group(table_75, "dmv-75")
    sample_features_75 = load_sample_feature(sampling_group_75, "dmv-75", table_100.data, standard_sample_75.sample)

    gen_dataset(queryset, table_75, table_100, histo, standard_sample_75, sampling_group_75, sample_features_75, "75", True)
    query_num_list = get_small_queryset("75")
    small_query_set = []
    query_num = 0
    for query in queryset["train"]:
        if query_num in query_num_list:
            small_query_set.append(query)
        query_num += 1
    queryset["tuning"] = small_query_set

    cal_append("80", queryset, table_75, table_100, histo)
    cal_append("85", queryset, table_75, table_100, histo)
    cal_append("90", queryset, table_75, table_100, histo)
    cal_append("95", queryset, table_75, table_100, histo)
    cal_append("100", queryset, table_75, table_100, histo)


    L.info("=========$$75$$=========")
    train_pkl = C.DATASET_PATH / "update" / "75-train.pkl"
    valid_pkl = C.DATASET_PATH / "update" / "75-valid.pkl"
    test_pkl = C.DATASET_PATH / "update" / "75-test.pkl"
    train_model("75", train_pkl, valid_pkl, C.DMV_TABLE_COL)
    model_path_75 = C.MODEL_PATH / "75.pkl"
    get_test_result(model_path_75, test_pkl, "75", C.DMV_TABLE_COL)

    # 80
    cal_append_result("80")
    cal_append_result("85")
    cal_append_result("90")
    cal_append_result("95")
    cal_append_result("100")


def cal_append_result(name):
    L.info(f"=========$${name}$$=========")
    model_path_75 = C.MODEL_PATH / "75.pkl"
    test_pkl = C.DATASET_PATH / "update" / f"{name}-test.pkl"
    valid_pkl = C.DATASET_PATH / "update" / "75-valid.pkl"

    test_only_sample = C.DATASET_PATH / "update" / f"{name}-sample-only-test.pkl"
    get_test_result(model_path_75, test_only_sample, name, C.DMV_TABLE_COL)

    # Tuning
    tuning_pkl = C.DATASET_PATH / "update" / f"{name}-tuning.pkl"
    tuning_model(model_path_75, name, tuning_pkl, valid_pkl, C.DMV_TABLE_COL)
    model_path_tuning = C.MODEL_PATH / f"{name}-tuning.pkl"
    get_test_result(model_path_tuning, test_only_sample, name, C.DMV_TABLE_COL)

    # Retrain
    train_pkl = C.DATASET_PATH / "update" / f"{name}-train.pkl"
    valid_pkl = C.DATASET_PATH / "update" / f"{name}-train.pkl"
    
    train_model(name, train_pkl, valid_pkl, C.DMV_TABLE_COL)
    model_path = C.MODEL_PATH / f"{name}.pkl"
    get_test_result(model_path, test_pkl, name, C.DMV_TABLE_COL)
