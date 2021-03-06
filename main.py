import os
import shutil
import argparse

import constants as C
from utils.log import Log
from GenDataset.multi.gen import gen_multi_dataset
from GenDataset.single.gen import gen_single_dataset
from GenDataset.single.workload.gen_workload import generate_workload
from model.train import train_model
from model.test import get_test_result


L = Log(__name__).get_logger()


def check_file(path):
    if not os.path.exists(path):
        L.warn(f"{path} not exist, generate first!")
        return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default="test", help="genload, gendata, train, test or clean")
    parser.add_argument('--table', type=str, default="single", help="multi or single")
    parser.add_argument("--dataset", type=str, default="power", help="the dir name of csv files")
    parser.add_argument("--workload", type=str, default="generate", help="the dir name of sql file")
    parser.add_argument("--testW", type=str, default="test", help="the dir name of sql file")
    arguments = parser.parse_args()

    if arguments.phase == "genload":
        if arguments.table == "single":
            queryset = generate_workload(seed=1, dataset=arguments.dataset, params=C.WORKLOAD_PARA)
        else:
            L.warn(f"{arguments.table}-table not implement!")

    elif arguments.phase == "gendata":
        L.info(f"Generating dataset for {arguments.table}-table {arguments.dataset} on {arguments.workload}")
        if arguments.table == "single":
            gen_single_dataset(arguments.dataset, arguments.workload)
        elif arguments.table == "multi" and arguments.dataset == "job-light":
            gen_multi_dataset(arguments.dataset, arguments.workload, C.JOB_TABLE_LIST,
                              C.JOB_JOIN_COL, C.JOB_BASE_TABLE, C.JOB_TABLE_DICT)
        else:
            L.warn(f"{arguments.table}-table {arguments.dataset} not implement!")

    elif arguments.phase == "train":
        L.info(f"Training for {arguments.table}-table {arguments.dataset} on {arguments.workload}")
        if arguments.table == "single" and arguments.workload == "generate":
            data_path = C.DATASET_PATH / f"{arguments.dataset}-{arguments.workload}-train.pkl"
            valid_path = C.DATASET_PATH / f"{arguments.dataset}-{arguments.workload}-valid.pkl"
            if arguments.dataset == "census":
                train_model(arguments.dataset, data_path, valid_path, 1.005, C.CENSUS_TABLE_COL, multi=False)
            elif arguments.dataset == "dmv":
                train_model(arguments.dataset, data_path, valid_path, 1.005, C.DMV_TABLE_COL, multi=False)
            elif arguments.dataset == "forest":
                train_model(arguments.dataset, data_path, valid_path, 1.005, C.FOREST_TABLE_COL, multi=False)
            elif arguments.dataset == "power":
                train_model(arguments.dataset, data_path, valid_path, 1.005, C.POWER_TABLE_COL, multi=False)
            else:
                L.warn("Training need schema, Please set them in constants!")
        elif arguments.table == "multi" and arguments.dataset == "job-light":
            data_path = C.DATASET_PATH / f"{arguments.dataset}-{arguments.workload}.pkl"
            if not os.path.exists(data_path):
                L.warn("Data not exist, generate first!")
            valid_path = C.DATASET_PATH / f"{arguments.dataset}-scale.pkl"
            train_model(arguments.dataset, data_path, valid_path, 1.05, C.JOB_COL_DIST, C.JOB_JOIN_COL, multi=True)
        else:
            L.warn("Training need schema, Please set them in constants!")

    elif arguments.phase == "test":
        if arguments.table == "single" and arguments.workload == "generate":
            data_path = C.DATASET_PATH / f"{arguments.dataset}-{arguments.workload}-test.pkl"
            model_path = C.MODEL_PATH / f"{arguments.dataset}.pkl"
            if arguments.dataset == "census":
                get_test_result(model_path, data_path, arguments.dataset, arguments.testW,
                                C.CENSUS_TABLE_COL, multi=False, value=1.005)
            elif arguments.dataset == "dmv":
                get_test_result(model_path, data_path, arguments.dataset, arguments.testW,
                                C.DMV_TABLE_COL, multi=False, value=1.005)
            elif arguments.dataset == "forest":
                get_test_result(model_path, data_path, arguments.dataset, arguments.testW,
                                C.FOREST_TABLE_COL, multi=False, value=1.005)
            elif arguments.dataset == "power":
                get_test_result(model_path, data_path, arguments.dataset, arguments.testW,
                                C.POWER_TABLE_COL, multi=False, value=1.005)
            else:
                L.warn("Testing need schema, Please set them in constants!")
        elif arguments.table == "multi" and arguments.dataset == "job-light":
            data_path = C.DATASET_PATH / f"{arguments.dataset}-{arguments.testW}.pkl"
            model_path = C.MODEL_PATH / f"{arguments.dataset}.pkl"
            get_test_result(model_path, data_path, arguments.dataset, arguments.workload,
                            C.JOB_COL_DIST, C.JOB_JOIN_COL, multi=True, value=1.05)
        else:
            L.warn("Testing need schema, Please set them in constants!")

    elif arguments.phase == "clean":
        L.info(f"Phase {arguments.phase}")
        shutil.rmtree(C.TEMP_ROOT)

    else:
        L.warn(f"Phase {arguments.phase} not implement!")
