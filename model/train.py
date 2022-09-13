import torch
import time
import numpy as np

import constants as C
from utils.log import Log
from utils.metrics import acc, confusion
from model.featureNew import load_data, load_data_multi, load_data_gbdt, load_data_multi_gbdt
from model.model import WideDeepNew
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix


def BCELoss_class_weighted(weights):

    def loss(input, target):
        input = torch.clamp(input, min=1e-7, max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)

    return loss


def train_one_epoch(model_m, loss_function, train_size, log, train_loader_t, valid_loader_t, optimizer_t):
    # train
    model_m.train()
    train_losses_t = []
    valid_losses_t = []
    train_acc_t = []
    valid_acc_t = []
    for i, (wide_feat, deep_feat, sql_feat, co_feat, label) in enumerate(train_loader_t):
        optimizer_t.zero_grad()
        output = model_m(wide_feat, deep_feat, sql_feat, co_feat)
        # label_arg_max = torch.argmax(label, dim=1)
        loss = loss_function(output, label)
        loss.backward()
        optimizer_t.step()
        train_losses_t.append(loss.item())
        acc_t = acc(output, label)
        train_acc_t.append(acc_t)
        if i % 1000 == 999:
            log.info('{:>6d}/{} train loss: {:.10f}, train acc: {:.10f}'.format((i + 1)
                     * 1024, train_size, np.mean(train_losses_t), np.mean(train_acc_t)))
    # evaluate
    model_m.eval()
    with torch.no_grad():
        for i, (wide_feat, deep_feat, sql_feat, co_feat, label) in enumerate(valid_loader_t):
            output = model_m(wide_feat, deep_feat, sql_feat, co_feat)
            # label_arg_max = torch.argmax(label, dim=1)
            acc_t = acc(output, label)
            loss = loss_function(output, label)
            valid_losses_t.append(loss.item())
            valid_acc_t.append(acc_t)
    return train_losses_t, train_acc_t, valid_losses_t, valid_acc_t


def train_one_epoch_new(model_m, loss_function, train_loader_t, valid_loader_t, optimizer_t):
    # train
    model_m.train()
    train_losses_t = []
    valid_losses_t = []
    train_acc_t = []
    valid_acc_t = []
    for i, (wide_feat, deep_feat, co_feat, sql_feat, label) in enumerate(train_loader_t):
        optimizer_t.zero_grad()
        output = model_m(wide_feat, deep_feat, co_feat, sql_feat)
        # label_arg_max = torch.argmax(label, dim=1)
        loss = loss_function(output, label)
        loss.backward()
        optimizer_t.step()
        train_losses_t.append(loss.item())
        acc_t = acc(output, label)
        train_acc_t.append(acc_t)
    # evaluate
    model_m.eval()
    with torch.no_grad():
        for i, (wide_feat, deep_feat, co_feat, sql_feat, label) in enumerate(valid_loader_t):
            output = model_m(wide_feat, deep_feat, co_feat, sql_feat)
            # label_arg_max = torch.argmax(label, dim=1)
            acc_t = acc(output, label)
            loss = loss_function(output, label)
            valid_losses_t.append(loss.item())
            valid_acc_t.append(acc_t)
    return train_losses_t, train_acc_t, valid_losses_t, valid_acc_t


def train_model(dataset_name, train_data_path, valid_data_path, table_col, join_col=None, multi=False):
    # 1. Load Data
    logger = Log(__name__, f"model-train-{dataset_name}").get_logger()
    if multi:
        train_label_df, train_loader, train_size, input_size = load_data_multi(
            train_data_path, "train", logger, C.DEVICE, table_col, join_col)
        valid_label_df, valid_loader, _, _ = load_data_multi(
            valid_data_path, "valid", logger, C.DEVICE, table_col, join_col)
    else:
        train_label_df, train_loader, train_size, input_size = load_data(
            train_data_path, "train", logger, C.DEVICE, table_col)
        valid_label_df, valid_loader, _, _ = load_data(valid_data_path, "valid", logger, C.DEVICE, table_col)
    logger.info("Model input: " + str(input_size))

    # 2. Setup Model
    start_time = time.time()
    torch.manual_seed(521)
    model = WideDeepNew(input_size)
    logger.info("Model established: " + str(model))
    model.to(C.DEVICE)
    class0_weight = max(input_size["class0"], input_size["class1"]) / input_size["class0"]
    class1_weight = max(input_size["class0"], input_size["class1"]) / input_size["class1"]
    weights = [class0_weight, class1_weight]
    class_weights = torch.FloatTensor(weights)
    logger.info(f"Loss Weights: {class_weights}")
    loss_function = BCELoss_class_weighted(class_weights)
    logger.info("Using BCELoss as loss function.")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger.info("Using Adam as optimization function.")

    # 3. Training Model
    logger.info("Begin the training of model.")
    for epoch in range(C.EPOCHES):
        train_losses, train_acc, valid_losses, valid_acc = train_one_epoch_new(
            model, loss_function, train_loader, valid_loader, optimizer)
        logger.info('epoch : {}, train loss : {:.4f}, train acc: {:.4f}, valid loss : {:.4f}, valid acc : {:.4f}'.format(
            epoch + 1, np.mean(train_losses), np.mean(train_acc), np.mean(valid_losses), np.mean(valid_acc)))
    train_time = (time.time() - start_time) / 60
    logger.info(f"Training Time of Neural Network: {train_time} min.")
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
    logger.info(metrics)
    # 4. Saving Model
    model_path = C.MODEL_PATH / f"{dataset_name}.pkl"
    logger.info("Saving Model.")
    torch.save(model, model_path)
    logger.info("Model saved in path " + str(model_path))


def train_gbdt(dataset_name, train_data_path, valid_data_path, table_col, join_col=None, multi=False):
    # 1. Load Data
    logger = Log(__name__, f"model-train-{dataset_name}").get_logger()
    if multi:
        train_X, train_y = load_data_multi_gbdt(train_data_path, "train", logger, table_col, join_col)
        valid_X, valid_y = load_data_multi_gbdt(valid_data_path, "valid", logger, table_col, join_col)
    else:
        train_X, train_y = load_data_gbdt(train_data_path, "train", logger, table_col)
        valid_X, valid_y = load_data_gbdt(valid_data_path, "valid", logger, table_col)

    # 2. Setup Model
    start_time = time.time()
    lgbmodel = lgb.LGBMClassifier(learning_rate=0.02, metric='cross_entropy', n_estimators=2000,
                                  num_leaves=128, is_unbalance=True, random_state=521)
    logger.info(str(lgbmodel).strip())
    lgbmodel.fit(train_X, train_y, eval_set=[(valid_X, valid_y)],
                 eval_metric=['accuracy'], early_stopping_rounds=20)
    train_time = (time.time() - start_time) / 60
    logger.info(f"Training Time of GBDT: {train_time} min.")

    pred_y = lgbmodel.predict(valid_X)
    accuracy = accuracy_score(pred_y, valid_y)
    logger.info(f"Accuracy = {accuracy}")
    cm = confusion_matrix(pred_y, valid_y)
    total = len(valid_y)
    logger.info(f"TP={round(cm[1, 1]/total, 3)}, FP={round(cm[1, 0]/total, 3)}")
    logger.info(f"TN={round(cm[0, 0]/total, 3)}, FN={round(cm[0, 1]/total, 3)}")
    # 4. Saving Model
    model_path = C.MODEL_PATH / f"{dataset_name}-gbdt.txt"
    logger.info("Saving Model.")
    lgbmodel.booster_.save_model(str(model_path))
    logger.info("Model saved in path " + str(model_path))


def train(dataset_name, train_data_path, valid_data_path, table_col, model="nn", join_col=None, multi=False):
    if model == "nn":
        train_model(dataset_name, train_data_path, valid_data_path, table_col, join_col, multi)
    else:
        train_gbdt(dataset_name, train_data_path, valid_data_path, table_col, join_col, multi)
