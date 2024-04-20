import json
import os
import shutil
from typing import Optional

import cv2
import uuid
from urllib.parse import quote, unquote

from sryolo.utils import label_utils
from sryolo.devtools import ultralytics_utils, os_utils
from ultralytics import YOLO


def get_label_studio_data_dir() -> str:
    """
    存放Label-Studio数据的根目录
    """
    return os.path.join(os_utils.get_work_dir(), 'label-studio')


def get_raw_images_dir() -> str:
    """
    原图的根目录
    """
    return os.path.join(get_label_studio_data_dir(), 'raw_images')


def get_annotations_dir() -> str:
    """
    标注的根目录 label-studio 自动同步Target的目录
    """
    return os.path.join(get_label_studio_data_dir(), 'annotations')


def get_tasks_dir() -> str:
    """
    标注任务的根目录 label-studio 自动同步Source的目录
    """
    path = os.path.join(get_label_studio_data_dir(), 'tasks')
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_sub_task_dir(v1_label):
    """
    标注任务的子目录
    """
    tasks_dir = get_tasks_dir()
    sub_task_dir = os.path.join(tasks_dir, v1_label)
    if not os.path.exists(sub_task_dir):
        os.mkdir(sub_task_dir)
    return sub_task_dir


def list_label_template(col: str):
    """
    显示label-studio需要用的分类模板
    :param col: 分类列
    """
    df = label_utils.read_label_csv()
    for v1 in df[col].drop_duplicates():
        print('<Label value="%s"/>' % v1)


def get_img_name_2_annotations() -> dict:
    """
    获取当前的标注
    key=图片文件名
    value=Label-Studio自动同步保存的标注
    """
    annotations_dir = get_annotations_dir()
    img_2_annotations = {}
    for annotation_file_name in os.listdir(annotations_dir):
        if annotation_file_name.find('.') > -1:
            continue
        annotation_file_path = os.path.join(annotations_dir, annotation_file_name)
        with open(annotation_file_path, 'r') as file:
            label_old = json.load(file)
            label_new = {
                'data': {},
                'annotations': []
            }
            label_new['data']['image'] = label_old['task']['data']['image']
            label_new['annotations'].append({
                'result': label_old['result']
            })
            file_path = unquote(label_new['data']['image'][21:])  # 21位开始是raw_images文件夹
            img_2_annotations[file_path[file_path.rfind('\\') + 1:]] = label_new

    return img_2_annotations


def fill_uid_black(img):
    """
    将截图的UID部分变成灰色
    :param img: 屏幕截图
    """
    lt = (30, 1030)
    rb = (200, 1080)
    cv2.rectangle(img, lt, rb, (114, 114, 114), -1)


def rename_raw_images(renew: bool = False) -> dict[str, str]:
    """
    对原图文件夹中的图片进行重名名 使用子文件夹名称做前缀
    :param renew: 是否全新 否的话只重命名未符合规范的 是的话全部重命名 使用前应该做好备份
    :return 返回重命名的映射关系
    """
    raw_img_dir = get_raw_images_dir()
    img_path_old_to_new = {}
    for prefix in os.listdir(raw_img_dir):
        sub_img_dir = os.path.join(raw_img_dir, prefix)
        if not os.path.isdir(sub_img_dir):
            continue

        to_name_list = []  # 需要重命名的图片
        existed_ids = set()  # 已经存在的id
        for img_name in os.listdir(sub_img_dir):
            if not img_name.endswith('.png'):
                continue
            if not renew and img_name.startswith(prefix):
                curr_id = img_name[-9:-4]
                existed_ids.add(int(curr_id))
            else:
                to_name_list.append(img_name)

        idx = 0
        for old_name in to_name_list:
            idx += 1
            while not renew and idx in existed_ids:
                idx += 1
            existed_ids.add(idx)

            new_name = '%s-%05d.png' % (prefix, idx)
            new_path = os.path.join(sub_img_dir, new_name)
            img_path_old_to_new[os.path.join(sub_img_dir, old_name)] = new_path

    old_path_list = list(img_path_old_to_new.keys())
    old_path_list.sort()
    for old_path in old_path_list:
        new_path = img_path_old_to_new[old_path]
        # print(old_path, new_path)
        shutil.move(old_path, new_path)

    return img_path_old_to_new


def generate_tasks_from_annotations(renew: bool = False):
    """
    从已有的标注文件中生成task 适合用于导入其他Label-Studio项目标注的数据
    并对原图文件夹中的图片进行重命名 按子文件夹名称做前缀

    """
    raw_img_dir = get_raw_images_dir()

    old_img_2_annotations = get_img_name_2_annotations()
    new_img_2_annotations = {}

    if old_name in old_img_2_annotations:
        new_annotations = old_img_2_annotations[old_name].copy()
        new_annotations['data']['image'] = '/data/local-files/?d=' + quote(
            new_path[new_path.find('raw_images'):])
        new_img_2_annotations[new_name] = new_annotations

    tasks_dir = get_tasks_dir()
    for img_name, annotations in new_img_2_annotations.items():
        new_task_path = os.path.join(tasks_dir, '%s.json' % img_name[:-4])
        with open(new_task_path, 'w') as file:
            json.dump(annotations, file, indent=4)


def get_with_task_case_ids():
    tasks_dir = get_tasks_dir()
    with_tasks_case_ids = set()
    for sub_task_dir_name in os.listdir(tasks_dir):
        sub_task_dir = os.path.join(tasks_dir, sub_task_dir_name)
        if not os.path.isdir(sub_task_dir):
            continue
        for task_json_name in os.listdir(sub_task_dir):
            if not task_json_name.endswith('.json'):
                continue
            with_tasks_case_ids.add(task_json_name[:-5])
    return with_tasks_case_ids


def generate_tasks_by_predictions(dataset_name: Optional[str] = None, train_name: Optional[str] = None, pt_name: str = 'best', use_label: str = 'v1'):
    """
    从raw_images中 找出还没有创建task的图片 进行预测并生成task
    """
    with_tasks_case_ids = get_with_task_case_ids()

    if dataset_name is not None and train_name is not None:
        dataset_label_idx_2_v1_label = ultralytics_utils.get_dataset_label_idx_2_v1_label(dataset_name, use_label)
        model_path = ultralytics_utils.get_train_model_path(dataset_name, train_name, pt_name)
        model = YOLO(model_path)
    else:
        model = None
    
    raw_images_dir = get_raw_images_dir()

    for sub_dir_name in os.listdir(raw_images_dir):
        sub_dir = os.path.join(raw_images_dir, sub_dir_name)
        if not os.path.isdir(sub_dir):
            continue
        sub_task_dir = get_sub_task_dir(sub_dir_name)
        if not os.path.exists(sub_task_dir):
            os.mkdir(sub_task_dir)
        for img_name in os.listdir(sub_dir):
            if not img_name.endswith('.png'):
                continue
            case_id = img_name[:-4]
            if case_id in with_tasks_case_ids:
                continue
            img_path = os.path.join(sub_dir, img_name)
            img = cv2.imread(img_path)
    
            task = {'data': {}}
            task['data']['image'] = '/data/local-files/?d=' + quote(img_path[img_path.find('raw_images'):])

            if model is not None:
                results = model.predict(img)
                result = results[0]

                if len(result.boxes) > 0:
                    task['predictions'] = [
                        {'model_version': f'{dataset_name}-{train_name}-{pt_name}', 'result': []}
                    ]

                for i in range(len(result.boxes.cls)):
                    cls = int(result.boxes.cls[i])
                    xywh = result.boxes.xywhn[i].tolist()
                    xyxy = result.boxes.xyxyn[i].tolist()
                    predict_result = {
                        'original_width': result.boxes.orig_shape[1],
                        'original_height': result.boxes.orig_shape[0],
                        'value': {
                            'x': xyxy[0] * 100,
                            'y': xyxy[1] * 100,
                            'width': xywh[2] * 100,
                            'height': xywh[3] * 100,
                            'rotation': 0,
                            'rectanglelabels': [
                                dataset_label_idx_2_v1_label[cls]
                            ]
                        },
                        'id': str(uuid.uuid4()),
                        'from_name': "label",
                        'to_name': "image",
                        'type': "rectanglelabels"
                    }
                    task['predictions'][0]['result'].append(predict_result)
    
            new_task_path = os.path.join(sub_task_dir, '%s.json' % case_id)
            with open(new_task_path, 'w') as file:
                json.dump(task, file, indent=4)
