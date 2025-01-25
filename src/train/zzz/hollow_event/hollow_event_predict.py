from ultralytics import YOLO

from train.devtools import os_utils, ultralytics_utils, label_studio_utils
from train.zzz.hollow_event import hollow_event_label

if __name__ == '__main__':
    model_name = 'yolov8s-736'
    pt_model_path = ultralytics_utils.get_train_model_path('zzz_hollow_event_2208', model_name, 'best', model_type='pt')
    pt_model = YOLO(pt_model_path)
    label_studio_utils.generate_tasks_by_predictions(
        project_dir=os_utils.get_path_under_work_dir('label_studio', 'zzz', 'hollow_event'),
        data_img_path_prefix='zzz\\hollow_event\\raw',
        model=pt_model,
        model_version=model_name,
        classes=hollow_event_label.get_labels_with_name(),
        # max_count=1
    )