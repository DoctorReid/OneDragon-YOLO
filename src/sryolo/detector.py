import time
import os
import urllib.request
from typing import Optional, List

import cv2
import numpy as np
import onnxruntime as ort

from sryolo.utils.detector_utils import multiclass_nms, xywh2xyxy, draw_detections
from sryolo.devtools import os_utils
from sryolo.utils.log_utils import log

_GH_PROXY_URL = 'https://mirror.ghproxy.com'
_MODEL_DOWNLOAD_PATH = 'https://github.com/DoctorReid/StarRail-YOLO/releases/download/model_download_test'


class StarRailYOLO:

    def __init__(self,
                 model_name: str = 'full-test-v0.onnx',
                 model_dir: Optional[str] = None,
                 conf_threshold: float = 0.7,
                 iou_threshold: float = 0.5,
                 cuda=False):
        """

        :param model_name:
        :param model_dir:
        :param conf_threshold:
        :param iou_threshold:
        :param cuda:
        """
        self.session: Optional[ort.InferenceSession] = None

        # 从模型中读取到的输入输出信息
        self.input_names: List[str] = []
        self.input_width: int = 0
        self.input_height: int = 0
        self.output_names: List[str] = []

        # 默认的阈值
        self.default_conf_threshold: float = conf_threshold
        self.default_iou_threshold: float = iou_threshold

        # 每次推理时使用的阈值
        self.conf_threshold: float = conf_threshold
        self.iou_threshold: float = iou_threshold

        # 加载模型
        model_path = self.get_model_path(model_dir, model_name)
        self.initialize_model(model_path, cuda)

    def get_model_path(self, model_dir: str, model_name: str) -> str:
        if model_dir is None:  # 默认使用本文件的目录
            model_dir = os.path.abspath(__file__)

        model_path = os.path.join(model_dir, model_name)

        if not os.path.exists(model_path):
            download = self.download_model(model_dir, model_name)
            if not download:
                raise Exception('模型下载失败 可手动下载模型')

        return model_path

    def download_model(self, model_path: str, model_name: str) -> bool:
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        url = f'{_GH_PROXY_URL}/{_MODEL_DOWNLOAD_PATH}/{model_name}'
        log.info('开始下载 %s %s', model_name, url)
        file_path = os.path.join(model_path, model_name)
        last_log_time = time.time()

        def log_download_progress(block_num, block_size, total_size):
            nonlocal last_log_time
            if time.time() - last_log_time < 1:
                return
            last_log_time = time.time()
            downloaded = block_num * block_size / 1024.0 / 1024.0
            total_size_mb = total_size / 1024.0 / 1024.0
            progress = downloaded / total_size_mb * 100
            log.info(f"正在下载 {model_name}: {downloaded:.2f}/{total_size_mb:.2f} MB ({progress:.2f}%)")

        try:
            file_name, response = urllib.request.urlretrieve(url, file_path, log_download_progress)
            log.info('下载完成 %s', model_name)
            return True
        except Exception:
            log.error('下载失败模型失败', exc_info=True)
            return False

    def initialize_model(self, path, cuda):
        availables = ort.get_available_providers()
        providers = ['CUDAExecutionProvider' if cuda else 'CPUExecutionProvider']
        if cuda and 'CUDAExecutionProvider' not in availables:
            log.error('机器未支持CUDA 使用CPU')
            providers = ['CPUExecutionProvider']

        log.info('加载模型 %s', path)
        self.session = ort.InferenceSession(
            path,
            providers=providers
        )
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image,
                       conf_threshold: Optional[float] = None,
                       iou_threshold: Optional[float] = 0.5):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        log.info(f"图片检测耗时: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        shape = model_inputs[0].shape
        self.input_height = shape[2]
        self.input_width = shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    model = StarRailYOLO(model_dir=os.path.join(os_utils.get_work_dir(), 'models'))