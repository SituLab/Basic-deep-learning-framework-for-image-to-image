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

（1）训练image-to-image任务

`  python main.py --running_name demo  `

（2）测试image-to-image任务

`  python main.py --running_name demo  --is_training 0 --is_testing 1`

（3）测试单张图像

`  python main.py --is_training 0 --img_path dataset/demo.png`

（4）数据集设置

`dataset/input/`存放输入的数据集；
`dataset/label`存放标签的数据集；
`dataset/test_input`存放测试输入的数据集；

（5）参数解释

`--running_name`：为每次训练提供一个运行名称，代码会创建相应名称的文件夹保存结果和日志。

`--is_train`：设置是否训练，默认训练；

`--is_test`：设置是否测试，默认测试；

`--img_path`：指定一张测试图像的路径；

（6）查看训练过程

`i`：log_demo.txt保存了此次训练所使用的配置信息和训练过程信息；

`ii`：weights/demo/best_model.pth保存了验证集loss最小的模型；

`iii`：results/demo/eval/保存了每一步训练时一个batch的推理结果；

（7）其他

在快速训练上，可以使用上述命令行的方法，如果需要细致开发，可以使用vscode或pycharm，使用编译器运行代码。


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
