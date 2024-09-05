# Basic deep learning framework for image-to-image

这个开发框架旨在帮助科研人员快速地实现图像到图像之间的模型开发。

## 目录
- [1模型开发](#1模型开发)
  - [1_1克隆项目到本地](#1_1克隆项目到本地)
  - [1_2深度学习开发](#1_2深度学习开发)
- [2环境配置](#2环境配置)
  - [2-1安装conda](#2-1安装conda)
  - [2-2安装pytorch](#2-2安装pytorch)


训练


## 1模型开发
### 1_1克隆项目到本地
（1）仓库右上角有个绿色‘code’按钮，下拉选择download zip。

（2）或者安装了git工具之后，在命令行运行下面指令：

`git clone https://github.com/SituLab/Basic-deep-learning-framework-for-image-to-image.git`

### 1_2深度学习开发
配置环境可以参考下一章节，若已安装，可跳过直接进行下面的开发。

注：该项目的代码经过优化，编写不少便于分析和调试的trick。

（1）训练自己的image-to-image任务

将网络的输入数据放在dataset文件夹的input文件夹下；将网络的标签数据放在dataset文件夹的label文件夹下；

打开cmd，切换到安装好的环境(conda )，运行`python main.py running_name`即可进行训练, 其中running_name是自定义一个此次运行的名称，代码会相应地按照该名称创建文件夹保存结果。

比如运行`python main.py demo`，会创建以demo为名称的文件夹

或者使用vscode或pycharm打开项目，使用编译器进行运行。

（2）查看训练过程

log_demo.txt文档保存了此次训练所使用的配置信息和训练过程信息；





## 2环境配置
建议有高配电脑，或者直接使用远程服务器已经配置好的环境。
### 2-1安装conda
annaconda，自带基础的python库，比较齐全，占用空间会比较大，网址：https://www.anaconda.com/download/
miniconda，纯净版conda命令软件，不自带库，需自行安装，占用空间小，网址：https://docs.anaconda.com/miniconda/

### 2-2安装pytorch
访问torch官网，直接通过指令进行安装。网址：https://pytorch.org/get-started/locally/
![image](https://github.com/user-attachments/assets/37652e77-f305-4814-88cc-d506ab77e1db)
比如：打开cmd，输入：

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
