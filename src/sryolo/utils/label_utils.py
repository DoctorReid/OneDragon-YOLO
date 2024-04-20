import os
import re

import pandas as pd

from sryolo.devtools import os_utils


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
