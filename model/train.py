import torch
import numpy as np

import constants as C
from utils.log import Log
from utils.metrics import acc
from model.feature import load_data, load_data_multi
from model.model import WideDeep


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


def train_model(dataset_name, train_data_path, valid_data_path, phi, table_col, join_col=None, multi=False):
    # 1. Load Data
    logger = Log(__name__, f"model-train-{dataset_name}").get_logger()
    if multi:
        train_label_df, train_loader, train_size, input_size = load_data_multi(
            train_data_path, "train", logger, C.DEVICE, phi, table_col, join_col)
        valid_label_df, valid_loader, _, _ = load_data_multi(
            valid_data_path, "valid", logger, C.DEVICE, phi, table_col, join_col)
    else:
        train_label_df, train_loader, train_size, input_size = load_data(
            train_data_path, "train", logger, C.DEVICE, phi, table_col)
        valid_label_df, valid_loader, _, _ = load_data(valid_data_path, "valid", logger, C.DEVICE, phi, table_col)
    logger.info("Model input: " + str(input_size))

    # 2. Setup Model
    torch.manual_seed(521)
    model = WideDeep(input_size)
    logger.info("Model established: " + str(model))
    model.to(C.DEVICE)
    loss_function = torch.nn.BCELoss()
    logger.info("Using BCELoss as loss function.")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger.info("Using Adam as optimization function.")

    # 3. Training Model
    logger.info("Begin the training of model.")
    for epoch in range(C.EPOCHES):
        train_losses, train_acc, valid_losses, valid_acc = train_one_epoch(
            model, loss_function, train_size, logger, train_loader, valid_loader, optimizer)
        logger.info('epoch : {}, train loss : {:.4f}, train acc: {:.4f}, valid loss : {:.4f}, valid acc : {:.4f}'.format(
            epoch + 1, np.mean(train_losses), np.mean(train_acc), np.mean(valid_losses), np.mean(valid_acc)))

    # 4. Saving Model
    model_path = C.MODEL_PATH / f"{dataset_name}.pkl"
    logger.info("Saving Model.")
    torch.save(model, model_path)
    logger.info("Model saved in path " + str(model_path))
