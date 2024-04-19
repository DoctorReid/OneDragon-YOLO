import os

import yaml
from ultralytics import settings

from sryolo.utils import os_utils, label_utils, str_utils


def get_ultralytics_dir() -> str:
    return os.path.join(os_utils.get_work_dir(), 'ultralytics')


def get_datasets_dir() -> str:
    return os.path.join(get_ultralytics_dir(), 'datasets')


def get_runs_dir() -> str:
    return os.path.join(get_ultralytics_dir(), 'runs')


def get_models_dir() -> str:
    return os.path.join(get_ultralytics_dir(), 'models')


def init_ultralytics_settings():
    settings.update({
        'datasets_dir': get_datasets_dir(),
        'weights_dir': get_models_dir(),
        'runs_dir': get_runs_dir()
    })


def get_base_model_path(model_name):
    return os.path.join(get_models_dir(), model_name)


def get_dataset_dir(dataset_name):
    return os.path.join(get_datasets_dir(), dataset_name)


def get_dataset_yaml_path(dataset_name):
    return os.path.join(get_dataset_dir(dataset_name), 'dataset.yaml')


def get_dataset_model_dir(dataset_name: str) -> str:
    """
    获取某个训练集对应的模型保存目录
    """
    return os.path.join(get_runs_dir(), dataset_name)


def get_train_model_path(dataset_name: str, train_name: str, pt_name: str = 'best') -> str:
    """
    获取一个训练模型的路径
    """
    return os.path.join(get_dataset_model_dir(dataset_name), train_name, 'weights', '%s.pt' % pt_name)


def get_dataset_label_idx_2_v1_label(dataset_name: str, use_label: str = 'v1') -> dict[int, str]:
    labels_df = label_utils.read_label_csv()
    dataset_label_2_v1_label = {}
    for _, row in labels_df.iterrows():
        dataset_label_2_v1_label[str_utils.without_cn_id_str(row[use_label])] = row['v1']

    with open(get_dataset_yaml_path(dataset_name), 'r') as file:
        dataset_config = yaml.safe_load(file)

    dataset_label_idx_2_v1_label = {}

    for i in range(100):
        if i not in dataset_config['names']:
            break
        dataset_label_idx_2_v1_label[i] = dataset_label_2_v1_label[dataset_config['names'][i]]

    return dataset_label_idx_2_v1_label
