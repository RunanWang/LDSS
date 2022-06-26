from math import log10


def cal_q_error(row_r):
    if row_r['label_GT_card'] == 0:
        a = 1
    elif row_r['label_est_card'] == 0:
        a = (row_r['label_est_card'] + 1) / (row_r['label_GT_card'] + 1)
    else:
        a = (row_r['label_est_card']) / (row_r['label_GT_card'])
    q_error = max(a, 1 / a)
    return q_error


def cal_error(est, gt):
    if gt == 0:
        a = 1
    elif est == 0:
        a = (est + 1) / (gt + 1)
    else:
        a = (est) / (gt)
    return log10(a)


def acc(y_pred, y_true):
    y_pred = y_pred > 0.5
    acc_mean = (y_pred == y_true).float().mean()
    return acc_mean.item()
