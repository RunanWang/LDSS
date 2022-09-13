import random
from math import log10

from utils.metrics import cal_q_error
from constants import STANDARD_SAMPLE_PAR


def gen_info_from_sample_set(to_choose_sample):
    info = {}
    info["ratio"] = 0
    info["time"] = 0
    info["est_card"] = 0
    info["GT_card"] = to_choose_sample[0]["GT_card"]
    info["std_qerror"] = to_choose_sample[0]["std_qerror"]
    info["std_time"] = to_choose_sample[0]["std_time"]
    est_card = 0
    for chosse_info in to_choose_sample:
        info["ratio"] += chosse_info["ratio"]
        info["time"] += chosse_info["time"]
        info["sample_name"] = "merge"
        est_card += chosse_info["ratio"] * chosse_info["est_card"]
    info["est_card"] = est_card / info["ratio"]
    info["q-error"] = cal_q_error(info["GT_card"], info["est_card"])
    return info


def gen_standard_sample_info(to_choose_sample):
    info = {}
    info["ratio"] = STANDARD_SAMPLE_PAR["ratio"]
    info["time"] = to_choose_sample[0]["std_time"]
    info["q-error"] = to_choose_sample[0]["std_qerror"]
    info["std_qerror"] = to_choose_sample[0]["std_qerror"]
    info["std_time"] = to_choose_sample[0]["std_time"]
    info["GT_card"] = to_choose_sample[0]["GT_card"]
    info["sample_name"] = "standard-sample"
    info["score"] = [1]
    info["est_card"] = to_choose_sample[0]["std_qerror"] * info["GT_card"]
    return info


def smallest_sample_wins(sample_dict):
    to_choose_sample = []
    # 先取所有满足阈值的查询
    for sample_unit, info_dict in sample_dict.items():
        if info_dict["score"][0] > 0.8:
            to_choose_sample.append(info_dict)
    # 这个查询模型预测都不满足阈值，则返回standard sample
    if len(to_choose_sample) == 0:
        for sample_unit, info_dict in sample_dict.items():
            to_choose_sample.append(info_dict)
        return gen_standard_sample_info(to_choose_sample)
    # 在满足阈值的查询中返回ratio最小的
    to_choose_sample.sort(key=lambda info: info["ratio"])
    min_ratio = 1
    min_sample_list = []
    for sample in to_choose_sample:
        if sample["ratio"] < min_ratio:
            min_ratio = sample["ratio"]
            min_sample_list = [sample]
        elif sample["ratio"] == min_ratio:
            min_sample_list.append(sample)
    return random.choice(min_sample_list)


def largest_sample_wins(sample_dict):
    to_choose_sample = []
    # 先取所有满足阈值的查询
    for sample_unit, info_dict in sample_dict.items():
        if info_dict["score"][0] > 0.8:
            to_choose_sample.append(info_dict)
    # 这个查询模型预测都不满足阈值，则返回standard sample
    if len(to_choose_sample) == 0:
        for sample_unit, info_dict in sample_dict.items():
            to_choose_sample.append(info_dict)
        return gen_standard_sample_info(to_choose_sample)
    # 在满足阈值的查询中返回ratio最大的
    to_choose_sample.sort(key=lambda info: info["ratio"])
    max_ratio = 0
    max_sample_list = []
    for sample in to_choose_sample:
        if sample["ratio"] > max_ratio:
            max_ratio = sample["ratio"]
            max_sample_list = [sample]
        elif sample["ratio"] == max_ratio:
            max_sample_list.append(sample)
    return random.choice(max_sample_list)


def most_prob_sample_wins(sample_dict):
    to_choose_sample = []
    # 先取所有满足阈值的查询
    for sample_unit, info_dict in sample_dict.items():
        if info_dict["score"][0] > 0.8:
            to_choose_sample.append(info_dict)
    # 这个查询模型预测都不满足阈值，则使用全部的抽样来预测
    if len(to_choose_sample) == 0:
        for sample_unit, info_dict in sample_dict.items():
            to_choose_sample.append(info_dict)
        return gen_standard_sample_info(to_choose_sample)
    # 有满足阈值的抽样，使用这些抽样进行预测
    to_choose_sample.sort(key=lambda info: info["score"][0])
    return to_choose_sample[-1]


def all_satisfy_sample_avg(sample_dict):
    to_choose_sample = []
    # 先取所有满足阈值的查询
    for sample_unit, info_dict in sample_dict.items():
        if info_dict["score"][0] > 0.8:
            to_choose_sample.append(info_dict)
    # 这个查询模型预测都不满足阈值，则使用全部的抽样来预测
    if len(to_choose_sample) == 0:
        for sample_unit, info_dict in sample_dict.items():
            to_choose_sample.append(info_dict)
        to_choose_sample = [(gen_standard_sample_info(to_choose_sample))]
    # 有满足阈值的抽样，使用这些抽样进行预测
    if len(to_choose_sample) >= 2:
        to_choose_sample.sort(key=lambda info: info["score"][0])
        to_choose_sample = to_choose_sample[-2:]
    return gen_info_from_sample_set(to_choose_sample)


def get_all_opt_sample(sample_dict):
    to_choose_sample = []
    # 先取所有满足阈值的查询
    for sample_unit, info_dict in sample_dict.items():
        to_choose_sample.append(info_dict)
    # 这个查询模型预测都不满足阈值，则使用全部的抽样来预测
    to_choose_sample.sort(key=lambda info: info["ratio"])
    to_choose_sample = to_choose_sample[:3]
    return gen_info_from_sample_set(to_choose_sample)


def opt_sample_wins(sample_dict):
    to_choose_sample = []
    # 先取所有满足阈值的查询
    for sample_unit, info_dict in sample_dict.items():
        if info_dict["label"] > 0.9:
            to_choose_sample.append(info_dict)
    # 这个查询模型预测都不满足阈值，则返回standard sample
    if len(to_choose_sample) == 0:
        for sample_unit, info_dict in sample_dict.items():
            to_choose_sample.append(info_dict)
        return gen_standard_sample_info(to_choose_sample)
    # 在满足阈值的查询中返回ratio最小的
    to_choose_sample.sort(key=lambda info: info["ratio"])
    return to_choose_sample[0]


def random_sample_wins(sample_dict):
    to_choose_sample = []
    # 先取所有满足阈值的查询
    for sample_unit, info_dict in sample_dict.items():
        to_choose_sample.append(info_dict)
    to_choose_sample.append(gen_standard_sample_info(to_choose_sample))
    return random.choice(to_choose_sample)


def standard_sample_wins(sample_dict):
    to_choose_sample = []
    # 先取所有满足阈值的查询
    for sample_unit, info_dict in sample_dict.items():
        to_choose_sample.append(info_dict)
    std_sample_info = gen_standard_sample_info(to_choose_sample)
    return std_sample_info


def has_small_sample(sample_dict):
    # 先取所有满足阈值的查询
    for sample_unit, info_dict in sample_dict.items():
        if info_dict["label"] > 0.9:
            return True
    return False


def get_best_sample_confidence(sample_dict):
    to_choose_sample = []
    all_sample = []
    total_confidence = 0
    choose_confidence = 0
    # 先取所有满足阈值的查询
    for sample_unit, info_dict in sample_dict.items():
        all_sample.append(info_dict)
        confidence = 3 + log10(info_dict["ratio"])
        total_confidence = total_confidence + confidence
        if info_dict["score"][0] > 0.75:
            to_choose_sample.append(info_dict)
            choose_confidence = choose_confidence + confidence
    all_sample.sort(key=lambda info: info["ratio"])
    # basic为所有抽样中ratio最大的
    basic = all_sample[-1]
    # 这个查询模型预测都不满足阈值，则返回ratio最大的
    if len(to_choose_sample) == 0:
        return basic
    # 遍历所有满足阈值的抽样，选出满足basic条件且ratio最小的，从中随机返回一个
    to_choose_sample.sort(key=lambda info: info["ratio"])
    if choose_confidence < total_confidence * 0.5:
        if len(to_choose_sample) > 1:
            return to_choose_sample[-1]
        else:
            return basic
    else:
        min_ratio = 1
        min_sample_list = []
        for sample in to_choose_sample:
            if sample["ratio"] < min_ratio:
                min_ratio = sample["ratio"]
                min_sample_list = [sample]
            elif sample["ratio"] == min_ratio:
                min_sample_list.append(sample)
        return random.choice(min_sample_list)
