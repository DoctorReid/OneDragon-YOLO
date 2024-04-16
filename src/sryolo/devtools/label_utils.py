import os

import pandas as pd

from sryolo.utils import os_utils


def read_label_csv() -> pd.DataFrame:
    return pd.read_csv(os.path.join(os_utils.get_work_dir(), 'labels.csv'))