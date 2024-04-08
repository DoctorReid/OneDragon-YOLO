# StarRail-YOLO

崩坏：星穹铁道 YOLO 目标检测

仅以此项目记录小白的第一次使用YOLO之旅。

# 开发环境说明

- Python版本 = 3.11.9
- CUDA版本 = 12.4 Windows x86_64 Version 11
- pytorch = 2.2.2 cuda=12.1 参考[官网](https://pytorch.org/get-started/locally/)安装
  - cuda12.1 `pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121`
  - cpu `pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu`
- 其他依赖 `pip install -r requirements.txt`

注意 `ultralytics` 自带依赖的pytorch不是cuda版本的，因此需要先安装pytorch或自己删除后重新安装。

`install.bat`是用来在本地目录下建立虚拟环境`.env/venv/`的，你也可以使用自己的环境。

# 训练

## 数据来源

图片来源于游戏中的智库，以及[一条龙]()脚本运行时的自动截图。



