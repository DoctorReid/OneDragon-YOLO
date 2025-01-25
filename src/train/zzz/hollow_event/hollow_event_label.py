import os
from typing import List

import pandas as pd

from train.devtools import os_utils


def get_label_df() -> pd.DataFrame:
    return pd.read_csv(os.path.join(
        os_utils.get_path_under_work_dir('labels', 'zzz'),
        'hollow_event.csv'
    ))


def get_labels_with_name() -> List[str]:
    label_df = get_label_df()
    result = []
    for index, row in label_df.iterrows():
        result.append('%04d-%s' % (row['label'], row['entry_name']))
    return result