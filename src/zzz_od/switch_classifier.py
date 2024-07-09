import os
import time
import urllib.request
import zipfile
from typing import Optional, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from utils.log_utils import log

class StarRailYOLO:

    def __init__(self,
                 model_name: str = 'yolov8n-1088-full-v1',
                 model_parent_dir_path: Optional[str] = None,
                 cuda: bool = False,
                 keep_result_seconds: float = 2
                 ):
        """
        :param model_name: 模型名称 在根目录下会有一个以模型名称创建的子文件夹
        :param model_parent_dir_path: 放置所有模型的根目录
        :param cuda: 是否启用CUDA
        :param keep_result_seconds: 保留多长时间的识别结果
        """
        self.session: Optional[ort.InferenceSession] = None

        # 从模型中读取到的输入输出信息
        self.input_names: List[str] = []
        self.onnx_input_width: int = 0
        self.onnx_input_height: int = 0
        self.output_names: List[str] = []

        # 分类
        self.idx_2_class: dict[int, DetectClass] = {}

        # 检测并下载模型
        model_dir_path = get_model_dir_path(model_parent_dir_path, model_name)
        # 加载模型
        self.load_model(model_dir_path, cuda)
        self.load_detect_classes(model_dir_path)

        # self.keep_result_seconds: float = keep_result_seconds
        # """保留识别结果的秒数"""
        # self.detect_result_history: List[DetectFrameResult] = []
        # """历史识别结果"""
        # self.last_detect_result: DetectFrameResult = None
        """最后一次识别结果"""

    def load_model(self, model_dir_path: str, cuda: bool) -> None:
        """
        加载模型
        :param model_dir_path: 存放模型的子目录
        :param cuda: 是否启用CUDA
        :return:
        """
        availables = ort.get_available_providers()
        providers = ['CUDAExecutionProvider' if cuda else 'CPUExecutionProvider']
        if cuda and 'CUDAExecutionProvider' not in availables:
            log.error('机器未支持CUDA 使用CPU')
            providers = ['CPUExecutionProvider']

        onnx_path = os.path.join(model_dir_path, 'model.onnx')
        log.info('加载模型 %s', onnx_path)
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers
        )
        self.get_input_details()
        self.get_output_details()

    def load_detect_classes(self, model_dir_path: str):
        """
        加载分类
        :param model_dir_path: model_dir_path: str
        :return:
        """
        csv_path = os.path.join(model_dir_path, 'labels.csv')
        labels_df = pd.read_csv(csv_path, encoding='utf-8')
        self.idx_2_class = {}
        for _, row in labels_df.iterrows():
            self.idx_2_class[row['idx']] = DetectClass(row['idx'], row['label'], row['cate'])

    def detect(self, image: MatLike,
               conf: float = 0.5,
               iou: float = 0.5,
               detect_time: Optional[float] = None) -> DetectFrameResult:
        """

        :param image: 使用 opencv 读取的图片 BGR通道
        :param conf: 置信度阈值
        :param iou: IOU阈值
        :param detect_time: 识别时间 应该是画面记录的时间 不传入时使用系统当前时间
        :return: 检测得到的目标
        """
        t1 = time.time()
        context = DetectContext(image, detect_time)
        context.conf = conf
        context.iou = iou

        input_tensor = self.prepare_input(context)
        t2 = time.time()

        outputs = self.inference(input_tensor)
        t3 = time.time()

        results = self.process_output(outputs, context)
        t4 = time.time()

        log.info(f'识别完毕 得到结果 {len(results)}个。预处理耗时 {t2 - t1:.3f}s, 推理耗时 {t3 - t2:.3f}s, 后处理耗时 {t4 - t3:.3f}s')

        self.record_frame_result(context, results)
        return self.last_detect_result

    def prepare_input(self, context: DetectContext) -> np.ndarray:
        """
        对检测图片进行处理 处理结果再用于输入模型
        参考 https://github.com/orgs/ultralytics/discussions/6994?sort=new#discussioncomment-8382661
        :param context: 上下文
        :return: 输入模型的图片 RGB通道
        """
        image = context.img
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 将图像缩放到模型的输入尺寸中较短的一边
        min_scale = min(self.onnx_input_height / context.img_height, self.onnx_input_width / context.img_width)

        # 未进行padding之前的尺寸
        context.scale_height = int(round(context.img_height * min_scale))
        context.scale_width = int(round(context.img_width * min_scale))

        # 缩放到目标尺寸
        if self.onnx_input_height != context.img_height or self.onnx_input_width != context.img_width:  # 需要缩放
            input_img = np.full(shape=(self.onnx_input_height, self.onnx_input_width, 3),
                                fill_value=114, dtype=np.uint8)
            scale_img = cv2.resize(rgb_img, (context.scale_width, context.scale_height), interpolation=cv2.INTER_LINEAR)
            input_img[0:context.scale_height, 0:context.scale_width, :] = scale_img
        else:
            input_img = rgb_img

        # 缩放后最后的处理
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor: np.ndarray):
        """
        图片输入到模型中进行推理
        :param input_tensor: 输入模型的图片 RGB通道
        :return: onnx模型推理得到的结果
        """
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_output(self, output, context: DetectContext) -> List[DetectObjectResult]:
        """
        :param output: 推理结果
        :param context: 上下文
        :return: 最终得到的识别结果
        """
        predictions = np.squeeze(output[0]).T

        # 按置信度阈值进行基本的过滤
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > context.conf, :]
        scores = scores[scores > context.conf]

        results: List[DetectObjectResult] = []
        if len(scores) == 0:
            return results

        # 选择置信度最高的类别
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # 提取Bounding box
        boxes = predictions[:, :4]  # 原始推理结果 xywh
        scale_shape = np.array([context.scale_width, context.scale_height, context.scale_width, context.scale_height])  # 缩放后图片的大小
        boxes = np.divide(boxes, scale_shape, dtype=np.float32)  # 转化到 0~1
        boxes *= np.array([context.img_width, context.img_height, context.img_width, context.img_height])  # 恢复到原图的坐标
        boxes = xywh2xyxy(boxes)  # 转化成 xyxy

        # 进行NMS 获取最后的结果
        indices = multiclass_nms(boxes, scores, class_ids, context.iou)

        for idx in indices:
            result = DetectObjectResult(rect=boxes[idx].tolist(),
                                        score=float(scores[idx]),
                                        detect_class=self.idx_2_class[int(class_ids[idx])]
                                        )
            results.append(result)

        return results

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        shape = model_inputs[0].shape
        self.onnx_input_height = shape[2]
        self.onnx_input_width = shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def record_frame_result(self, context: DetectContext, results: List[DetectObjectResult]) -> None:
        """
        记录本帧识别结果
        :param context: 识别上下文
        :param results: 识别结果
        :return: 组合结果
        """
        new_frame = DetectFrameResult(
                raw_image=context.img,
                results=results,
                detect_time=context.detect_time
            )
        self.last_detect_result = new_frame
        self.detect_result_history.append(new_frame)
        self.detect_result_history = [i for i in self.detect_result_history
                                      if context.detect_time - i.detect_time > self.keep_result_seconds]