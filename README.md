# OneDragon-YOLO

用于一条龙脚本的YOLO项目

# 开发环境说明

- Python版本 = 3.11.9
- CUDA版本 = 12.4 Windows x86_64 Version 11
- ultralytics = 参考[官网](https://docs.ultralytics.com/quickstart/#install-ultralytics)
  - cuda by conda `conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics`
  - cpu `pip install ultralytics`。注意本方式安装的pytorch不是cuda版本的，如果在这之后想用pip安装cuda的pytorch，需要自行删除后重新安装，参考下面。
  - cuda by pip 参考[官网](https://pytorch.org/get-started/locally/)安装(版本不一定对了) `pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121`
- 其他依赖 `pip install -r requirements.txt`

# 训练

## 数据来源

- 游戏中的智库
- [一条龙](https://github.com/DoctorReid/StarRailOneDragon)脚本运行时的自动截图
- 手动截图补充

## 分类

[表格](labels/sr/labels.csv)

# 其他 

## 我的探索过程

1. [第一次接触YOLO](notebook/experiments/01-first-trial/first.ipynb)
2. []