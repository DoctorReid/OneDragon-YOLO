import os
import shutil
from typing import Optional, Tuple

import yaml
from ultralytics import YOLO
from ultralytics import settings

from train.devtools import os_utils


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


def get_dataset_images_dir(dataset_name):
    return os.path.join(get_dataset_dir(dataset_name), 'images')


def get_dataset_labels_dir(dataset_name):
    return os.path.join(get_dataset_dir(dataset_name), 'labels')


def get_dataset_labels_bk_dir(dataset_name):
    return os.path.join(get_dataset_dir(dataset_name), 'labels_bk')


def get_dataset_yaml_path(dataset_name):
    return os.path.join(get_dataset_dir(dataset_name), 'dataset.yaml')


def get_dataset_model_dir(dataset_name: str) -> str:
    """
    获取某个训练集对应的模型保存目录
    """
    return os.path.join(get_runs_dir(), dataset_name)


def get_train_model_path(dataset_name: str, train_name: str, model_name: str = 'best', model_type: str = 'pt') -> str:
    """
    获取一个训练模型的路径
    """
    return os.path.join(get_dataset_model_dir(dataset_name), train_name, 'weights', '%s.%s' % (model_name, model_type))


def get_export_save_dir(dataset_name: str, model_name: str) -> str:
    """
    获取导出模型的用来保存的文件夹
    :param model_name:
    :return:
    """
    return os_utils.get_path_under_work_dir('models', dataset_name, model_name)


def export_model(dataset_name: str,
                 train_name: str = 'train',
                 model_name: str = 'best',
                 save_name: Optional[str] = None,
                 imgsz: Tuple[int, int] = (384, 640)):
    """
    导出模型
    1. 在models文件夹下创建子文件夹
    2. 保存 onnx模型 和 对应的标签csv 到子文件夹中
    :param dataset_name: 导出模型用的数据集
    :param train_name: 导出模型用的训练名
    :param model_name: 导出模型的名称
    :param save_name: 最终保存的模型名
    :return:
    """
    pt_model_path = get_train_model_path(dataset_name, train_name, model_name, model_type='pt')
    pt_model = YOLO(pt_model_path)
    pt_model.export(format='onnx', imgsz=imgsz)

    if save_name is None:
        save_name = train_name

    export_dir = get_export_save_dir(dataset_name, save_name)
    onnx_model_path = get_train_model_path(dataset_name, train_name, model_name, model_type='onnx')
    save_model_path = os.path.join(export_dir, 'model.onnx')

    shutil.move(onnx_model_path, save_model_path)

    yml_path = os.path.join(get_dataset_dir(dataset_name), 'dataset.yaml')
    with open(yml_path, 'r', encoding='utf-8') as file:
        yml_data = yaml.safe_load(file)

    labels_csv_path = os.path.join(export_dir, 'labels.csv')
    with open(labels_csv_path, 'w', encoding='utf-8') as file:
        file.write('idx,label\n')
        label_data = yml_data.get('names', {})
        for idx, label in label_data.items():
            file.write('%d,%s\n' % (idx, label))


def export_cls_model(
        dataset_name: str,
        train_name: str = 'train',
        model_name: str = 'best',
        save_name: Optional[str] = None,
        imgsz: Tuple[int, int] = (384, 640)
):
    pt_model_path = get_train_model_path(dataset_name, train_name, model_name, model_type='pt')
    pt_model = YOLO(pt_model_path)
    pt_model.export(format='onnx', imgsz=imgsz)

    if save_name is None:
        save_name = train_name

    export_dir = get_export_save_dir(dataset_name, save_name)
    onnx_model_path = get_train_model_path(dataset_name, train_name, model_name, model_type='onnx')
    save_model_path = os.path.join(export_dir, 'model.onnx')

    shutil.move(onnx_model_path, save_model_path)

    train_dataset_dir = os.path.join(get_dataset_dir(dataset_name), 'train')
    cls_data = []
    for cls_dir in os.listdir(train_dataset_dir):
        cls_data.append(cls_dir.split('-'))

    labels_csv_path = os.path.join(export_dir, 'labels.csv')
    with open(labels_csv_path, 'w', encoding='utf-8') as file:
        file.write('idx,label\n')
        for cls_item in cls_data:
            file.write(f'{cls_item[0]},{cls_item[1]}\n')



if __name__ == '__main__':
    # export_model(
    #     dataset_name='zzz_hollow_event_2208',
    #     train_name='yolov8n-640',
    # )

    export_cls_model(
        dataset_name='zzz_dodge_raw',
        train_name = 'train',
        imgsz = 640,
        model_name = 'best',
    )