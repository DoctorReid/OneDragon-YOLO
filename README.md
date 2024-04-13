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

- 游戏中的智库
- [一条龙](https://github.com/DoctorReid/StarRailOneDragon)脚本运行时的自动截图
- 手动截图补充

## 分类

|分类|描述|
|---|---|
|enemy-humanoid|敌人-人形|
|enemy-sphericity|敌人-球形|
|enemy-flying-wind|敌人-带飞翼|
|enemy-biped|敌人-两足动物|
|enemy-quadruped|敌人-四足动物|
|enemy-machine|敌人-机器类|
|npc-herta|NPC-黑塔|
|sim-entry-not-active|模拟宇宙-下层入口-未激活|
|sim-entry-active|模拟宇宙-下层入口-已激活|
|sim-event|模拟宇宙-事件牌|
|sim-reward|模拟宇宙-沉浸奖励装置|
|destroy-technique|可破坏物-秘技点|

具体分类见 [表格](分类表格.md)，数据标注使用[Label-Studio](https://github.com/DoctorReid/LabelStudio-windows-pip)

# 其他 

## 我的探索过程

1. [第一次接触YOLO](notebook/experiments/01-first-trial/first.ipynb)
2. []