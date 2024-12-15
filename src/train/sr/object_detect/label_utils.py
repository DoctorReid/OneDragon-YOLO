import os
import re
from typing import List

import pandas as pd

from train.devtools import os_utils


def read_label_csv() -> pd.DataFrame:
    return pd.read_csv(os.path.join(os_utils.get_work_dir(), 'labels.csv'))


def remove_cn_in_label(label: str) -> str:
    """
    将标签消除中文
    中文在部分库中支持不好
    :param label: 原标签名
    :return:
    """
    return re.sub(r'[^a-zA-Z0-9\-]+', '', label).replace('--', '-')


def read_sim_uni_objects() -> List[str]:
    """
    模拟宇宙中会出现的内容
    """
    df = pd.read_csv(os.path.join(os_utils.get_work_dir(), 'sim_uni.csv'))
    return df['name'].values


def read_world_patrol_objects() -> List[str]:
    """
    锄大地中需要识别的内容
    """
    df = pd.read_csv(os.path.join(os_utils.get_work_dir(), 'world_patrol.csv'))
    return df['name'].values

