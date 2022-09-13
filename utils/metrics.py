from math import log10
import torch


def cal_q_error(gt_card, est_card):
    if gt_card == 0:
        a = 1
    elif est_card == 0:
        a = (est_card + 1) / (gt_card + 1)
    else:
        a = (est_card) / (gt_card)
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


def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
    prediction = prediction > 0.5
    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives
