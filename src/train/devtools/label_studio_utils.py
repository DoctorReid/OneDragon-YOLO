import json
import os
import shutil
import uuid
from typing import List, Optional
from urllib.parse import quote, unquote

import cv2
import pandas as pd
from ultralytics import YOLO


def get_label_studio_data_dir() -> str:
    """
    存放Label-Studio数据的根目录
    """
    return os.path.join(os_utils.get_work_dir(), 'label-studio')


def get_raw_images_dir(project_dir: str) -> str:
    """
    原图的根目录
    """
    return os.path.join(project_dir, 'raw')


def get_annotations_dir() -> str:
    """
    标注的根目录 label-studio 自动同步Target的目录
    """
    return os.path.join(get_label_studio_data_dir(), 'annotations')


def get_tasks_dir(project_dir: str) -> str:
    """
    标注任务的根目录 label-studio 自动同步Source的目录
    """
    path = os.path.join(project_dir, 'tasks')
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_sub_task_dir(project_dir: str, label_name: str):
    """
    标注任务的子目录
    """
    tasks_dir = get_tasks_dir(project_dir)
    sub_task_dir = os.path.join(tasks_dir, label_name)
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


def get_img_name_2_annotations(project_dir: str,
                               old_img_path_prefix: Optional[str] = None,
                               new_img_path_prefix: Optional[str] = None,) -> dict:
    """
    获取当前的标注
    key=图片文件名
    value=Label-Studio自动同步保存的标注
    :param project_dir:
    :param old_img_path_prefix: 旧的图片路径根目录
    :param new_img_path_prefix: 新的图片路径根目录
    """
    annotations_dir = os.path.join(project_dir, 'annotations')
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
            label_new['id'] = label_old['id']
            label_new['data']['image'] = label_old['task']['data']['image']
            label_new['annotations'].append({
                'result': label_old['result']
            })

            common_image_file_prefix = '/data/local-files/?d='
            file_path = unquote(label_new['data']['image'].replace(common_image_file_prefix, ''))

            # 更正图片路径
            if old_img_path_prefix is not None and new_img_path_prefix is not None:
                file_path = file_path.replace(old_img_path_prefix, new_img_path_prefix)
                label_new['data']['image'] = f'{common_image_file_prefix}{quote(file_path)}'

            img_2_annotations[file_path[file_path.rfind('\\') + 1:]] = label_new

    return img_2_annotations


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


def generate_tasks_from_annotations(
        project_dir: str,
        old_img_path_prefix: Optional[str] = None,
        new_img_path_prefix: Optional[str] = None,):
    """
    从已有的标注文件中生成task 适合用于导入其他Label-Studio项目标注的数据
    """
    img_2_annotations = get_img_name_2_annotations(
        project_dir,
        old_img_path_prefix=old_img_path_prefix,
        new_img_path_prefix=new_img_path_prefix
    )

    tasks_dir = get_tasks_dir(project_dir)
    for img_name, annotations in img_2_annotations.items():
        new_task_path = os.path.join(tasks_dir, '%s.json' % img_name[:-4])
        with open(new_task_path, 'w') as file:
            json.dump(annotations, file, indent=4)


def get_with_task_case_ids(project_dir: str):
    tasks_dir = get_tasks_dir(project_dir)
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


def generate_tasks_by_predictions(
        project_dir: str, data_img_path_prefix: str,
        model: YOLO, model_version: str, classes: List[str],
        max_count: Optional[int] = None
) -> None:
    """
    从raw_images中 找出还没有创建task的图片 进行预测并生成task
    :param project_dir: 项目目录
    :param model: 模型
    :param model_version: 模型版本
    :param classes: 类别
    :param max_count: 最大生成数量
    """
    with_tasks_case_ids = get_with_task_case_ids(project_dir)
    with_tasks_case_ids = {}
    with_annotations_case_ids = [i[:-4] for i in get_img_name_2_annotations(project_dir).keys()]
    raw_images_dir = get_raw_images_dir(project_dir)

    cnt = 0
    for sub_dir_name in os.listdir(raw_images_dir):
        sub_dir = os.path.join(raw_images_dir, sub_dir_name)
        if not os.path.isdir(sub_dir):
            continue
        sub_task_dir = get_sub_task_dir(project_dir, sub_dir_name)
        if not os.path.exists(sub_task_dir):
            os.mkdir(sub_task_dir)
        for img_name in os.listdir(sub_dir):
            if not img_name.endswith('.png'):
                continue
            case_id = img_name[:-4]
            if case_id in with_tasks_case_ids:
                continue
            if case_id in with_annotations_case_ids:
                continue
            img_path = os.path.join(sub_dir, img_name)
            img = cv2.imread(img_path)
    
            task = {'data': {}}
            task['data']['image'] = '/data/local-files/?d=' + quote(img_path[img_path.find(data_img_path_prefix):])

            if model is not None:
                results = model.predict(img)
                result = results[0]

                if len(result.boxes) > 0:
                    task['predictions'] = [
                        {'model_version': model_version, 'result': []}
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
                            'rectanglelabels': [classes[cls]]
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
            cnt += 1
            if max_count is not None and cnt >= max_count:
                return


def get_img_name_2_path(raw_dir: str) -> dict[str, str]:
    """
    获取一个项目中 图片名字对应的图片路径
    """
    result = {}
    sub_dir_list = os.listdir(raw_dir)
    for sub_dir_name in sub_dir_list:
        sub_dir_path = os.path.join(raw_dir, sub_dir_name)
        if not os.path.isdir(sub_dir_path):
            continue

        img_list = os.listdir(sub_dir_path)
        for img_name in img_list:
            if not img_name.endswith('.png'):
                continue
            img_path = os.path.join(sub_dir_path, img_name)
            result[img_name[:-4]] = img_path

    return result


def print_labeling_interface(label_df: pd.DataFrame, label_col: str, class_col: str):
    """
    根据类别 输入 Labeling Interface 中的配置
    :param label_df: 类别数据
    :param label_col: 类别ID的列名 通常是4位数字
    :param class_col: 类别名称列名 通常是中文名称
    """
    for index, row in label_df.iterrows():
        print('<Label value="%04d-%s" />' % (row[label_col], row[class_col]))

def create_sub_dir_in_raw(project_dir: str, label_df: pd.DataFrame, label_col: str, class_col: str) -> None:
    """
    按照分类 创建子文件夹
    :param project_dir: 项目目录
    :param label_df: 类别数据
    :param label_col: 类别ID的列名 通常是4位数字
    :param class_col: 类别名称列名 通常是中文名称
    """
    raw_dir = get_raw_images_dir(project_dir)

    for index, row in label_df.iterrows():
        sub_dir_name = '%04d-%s' % (row[label_col], row[class_col])
        sub_dir_path = os.path.join(raw_dir, sub_dir_name)
        if not os.path.exists(sub_dir_path):
            os.mkdir(sub_dir_path)


def rename_file_in_raw_sub_dir(project_dir: str) -> None:
    """
    对原图的文件夹进行重命名
    :param project_dir: 项目目录
    """
    raw_dir = get_raw_images_dir(project_dir)
    for sub_dir_name in os.listdir(raw_dir):
        sub_dir = os.path.join(raw_dir, sub_dir_name)
        if not os.path.isdir(sub_dir):
            continue

        max_idx: int = 0
        for file_name in os.listdir(sub_dir):
            if not file_name.endswith('.png'):
                continue
            if file_name.startswith(sub_dir_name):
                idx = int(file_name[-8:-4])
                max_idx = max(idx, max_idx)
        max_idx += 1

        for file_name in os.listdir(sub_dir):
            if not file_name.endswith('.png'):
                continue

            if file_name.startswith(sub_dir_name):
                continue

            old_file_path = os.path.join(sub_dir, file_name)
            new_file_path = os.path.join(sub_dir, '%s-%04d.png' % (sub_dir_name, max_idx))
            os.rename(old_file_path, new_file_path)
            max_idx += 1


if __name__ == '__main__':
    from train.devtools import os_utils, ultralytics_utils
    from train.zzz.hollow_event import label_utils
    model_name = 'yolov8s-736'
    pt_model_path = ultralytics_utils.get_train_model_path('zzz_hollow_event_2208', model_name, 'best', model_type='pt')
    pt_model = YOLO(pt_model_path)
    generate_tasks_by_predictions(
        project_dir=os_utils.get_path_under_work_dir('label_studio', 'zzz', 'hollow_event'),
        data_img_path_prefix='zzz\\hollow_event\\raw',
        model=pt_model,
        model_version=model_name,
        classes=label_utils.get_labels_with_name(),
        # max_count=1
    )
    # generate_tasks_from_annotations(
    #     project_dir=os_utils.get_path_under_work_dir('label_studio', 'zzz', 'hollow_event'),
    # )