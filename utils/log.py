# !/usr/bin/python
# -*- coding:utf-8 -*-
import time
import logging
import pandas as pd
from constants import LOG_ROOT


class Log(object):

    def __init__(self, logger=None, filename="log"):
        # 创建一个logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)
        # 创建一个handler，用于写入日志文件
        self.log_time = time.strftime("%y%m%d-%H%M")
        file_dir = LOG_ROOT
        file_name = f"{filename}.log"
        self.log_name = file_dir / file_name
        fh = logging.FileHandler(self.log_name, 'a', encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S", )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # 关闭打开的文件
        fh.close()
        ch.close()

    def get_logger(self):
        return self.logger


def data_describe_log(df_name: str, df: pd.DataFrame, log: logging.Logger):
    log.info("Below is data describe of " + df_name)
    log.info("Count:    " + str(df.count()[0]))
    log.info("Mean :    " + str(df.mean()[0]))
    log.info("Std  :    " + str(df.std()[0]))
    log.info("Min  :    " + str(df.min()[0]))
    log.info("10%  :    " + str(df.quantile(0.1)[0]))
    log.info("50%  :    " + str(df.quantile(0.5)[0]))
    log.info("95%  :    " + str(df.quantile(0.95)[0]))
    log.info("99%  :    " + str(df.quantile(0.99)[0]))
    log.info("Max  :    " + str(df.max()[0]))
