# mediapipe_ybt
基于MediaPipe人体动作识别模型实现的Y Balance Test智能测试系统。

The Y Balance Test intelligent testing system based on MediaPipe.

本项目基于@MichistaLin的mediapipe-Fitness-counter项目开发，训练算法与视频检测逻辑与原项目一致，项目地址：https://github.com/MichistaLin/mediapipe-Fitness-counter


Y Balance Test是运动医学与运动康复领域的一个测试动作，用于评判受试者的下肢动态平衡能力，进一步了解：https://www.physio-pedia.com/Y_Balance_Test

在原项目动作分类检测的基础上增加了ybt.py文件用于取到YBT测试中所需要的身体关键点，并且进行计算输出测试者下肢长度及伸出脚伸出距离。

本项目main.py为测试视频的检测输出程序，不包含模型训练，模型训练程序算法可参考原项目。
