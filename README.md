# Basic deep learning framework for image-to-image

这个开发框架旨在帮助科研人员快速地实现图像到图像之间的模型开发。

# 1. 环境配置
建议有高配电脑，或者直接使用远程服务器已经配置好的环境。
## （1）安装annaconda或miniconda
annaconda，自带基础的python库，比较齐全，占用空间会比较大，网址：https://www.anaconda.com/download/
miniconda，纯净版conda命令软件，不自带库，需自行安装，占用空间小，网址：https://docs.anaconda.com/miniconda/
## （2）安装pytorch环境
访问torch官网，直接通过指令进行安装。网址：https://pytorch.org/get-started/locally/
![image](https://github.com/user-attachments/assets/37652e77-f305-4814-88cc-d506ab77e1db)
比如：打开cmd，输入：

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

# 2. 模型开发
## （1）将整个库下载或克隆到本地
仓库右上角有个绿色‘code’按钮，下拉选择clone到本地
