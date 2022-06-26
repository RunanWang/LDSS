import random
from math import log10


def get_best_sample(sample_dict):
    to_choose_sample = []
    # 先取所有满足阈值的查询
    for sample_unit, info_dict in sample_dict.items():
        if info_dict["score"][0] > 0.75:
            to_choose_sample.append(info_dict)
    # 这个查询模型预测都不满足阈值，则返回ratio最大的
    if len(to_choose_sample) == 0:
        for sample_unit, info_dict in sample_dict.items():
            to_choose_sample.append(info_dict)
        to_choose_sample.sort(key=lambda info: info["ratio"])
        return to_choose_sample[-1]
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


def get_best_sample_max(sample_dict):
    to_choose_sample = []
    all_sample = []
    # 先取所有满足阈值的查询
    for sample_unit, info_dict in sample_dict.items():
        all_sample.append(info_dict)
        if info_dict["score"][0] > 0.75:
            to_choose_sample.append(info_dict)
    all_sample.sort(key=lambda info: info["ratio"])
    # basic为所有抽样中ratio最大的
    basic = all_sample[-1]
    # 这个查询模型预测都不满足阈值，则返回ratio最大的
    if len(to_choose_sample) == 0 or basic not in to_choose_sample:
        return basic
    # 遍历所有满足阈值的抽样，选出满足basic条件且ratio最小的，从中随机返回一个
    min_ratio = 1
    min_sample_list = []
    to_choose_sample.sort(key=lambda info: info["ratio"])
    for sample in to_choose_sample:
        if sample["score"][0] > basic["score"][0] + 0.5:
            continue
        else:
            if sample["ratio"] < min_ratio:
                min_ratio = sample["ratio"]
                min_sample_list = [sample]
            elif sample["ratio"] == min_ratio:
                min_sample_list.append(sample)
    if len(min_sample_list) != 0:
        return random.choice(min_sample_list)
    # 都不满足阈值，返回basic
    return basic


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


def get_optimal_best_sample(sample_dict, value=1.005):
    to_choose_sample = []
    # 先取所有满足阈值的查询
    for sample_unit, info_dict in sample_dict.items():
        if info_dict["q-error"] < value:
            to_choose_sample.append(info_dict)
    # 这个查询模型预测都不满足阈值，则返回ratio最大的
    if len(to_choose_sample) == 0:
        for sample_unit, info_dict in sample_dict.items():
            to_choose_sample.append(info_dict)
        to_choose_sample.sort(key=lambda info: info["ratio"])
        return to_choose_sample[-1]
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
    return min_sample_list[0]
