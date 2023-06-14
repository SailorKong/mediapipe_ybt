# encoding utf-8
"""
 @Author: Sailor
 @FileName: main.py
 @DateTime: 2023.01.31 10:36
 @SoftWare: PyCharm
"""

# 对测试集视频进行检测
import videoprocess as vp
import trainingsetprocess as tp
import os

videos_path = "test_videos/"
i = 0
if __name__ == '__main__':
    for poses in os.listdir(videos_path):
        flag = f'{poses[1]}_{poses}'
        for item in os.listdir(videos_path+poses):
            item_path = f'{videos_path}{poses}/{item}'
            tp.trainset_process(flag)
            i += 1
            vp.video_process(item_path, flag, item)
            print('已处理 ', i, ' 个视频')