# encoding utf-8
"""
 @Author: Sailor
 @FileName: ybt.py
 @DateTime: 2023.02.10 09:35
 @SoftWare: PyCharm
"""
import csv
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

class YbtHelper(object):
    def __init__(self, ybt_data_out_folder, class_name, file_name):
        self._ybt_data_out_folder = ybt_data_out_folder
        self._class_name = class_name
        self._file_name = file_name
        self._landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

    def get_dist(self, point_a, point_b):
        """
        通过三维坐标得到欧式距离
            :param point_a: 第一个点的坐标，list类
            :param point_b: 第二个点的坐标，list类
            :return: 两点间距离 单位cm
        """
        # 转化为array类型便于计算
        point_a = np.array(point_a)
        point_b = np.array(point_b)

        # 两点间的欧式距离，单位cm
        dist = np.abs((np.power(point_a[0] - point_b[0], 2)
                       + np.power(point_a[1] - point_b[1], 2)
                       + np.power(point_a[2] - point_b[2], 2)) ** 0.5) * 100

        # 取小数后两位
        dist = np.around(dist, 2)

        return dist

    def get_reach(self, landmarks):
        """
        获取伸出脚到达的距离
        :param landmarks: pose检测得到的真实世界三维关键点坐标
        :return: 当前帧画面中的伸出脚到达距离
        """
        left_foot_index = [landmarks[self._landmark_names.index('left_foot_index')].x,
                           landmarks[self._landmark_names.index('left_foot_index')].y,
                           landmarks[self._landmark_names.index('left_foot_index')].z]

        right_foot_index = [landmarks[self._landmark_names.index('right_foot_index')].x,
                            landmarks[self._landmark_names.index('right_foot_index')].y,
                            landmarks[self._landmark_names.index('right_foot_index')].z]

        reach_dist = self.get_dist(left_foot_index, right_foot_index)

        return reach_dist

    def get_leg(self, landmarks):
        """
        获取下肢长度
        :param landmarks:pose检测得到的真实世界三维关键点坐标
        :return:当前帧画面中的双侧下肢中的最长一侧距离
        """
        left_hip = [landmarks[self._landmark_names.index('left_hip')].x,
                    landmarks[self._landmark_names.index('left_hip')].y,
                    landmarks[self._landmark_names.index('left_hip')].z]

        left_heel = [landmarks[self._landmark_names.index('left_heel')].x,
                     landmarks[self._landmark_names.index('left_heel')].y,
                     landmarks[self._landmark_names.index('left_heel')].z]

        right_hip = [landmarks[self._landmark_names.index('right_hip')].x,
                     landmarks[self._landmark_names.index('right_hip')].y,
                     landmarks[self._landmark_names.index('right_hip')].z]

        right_heel = [landmarks[self._landmark_names.index('right_heel')].x,
                      landmarks[self._landmark_names.index('right_heel')].y,
                      landmarks[self._landmark_names.index('right_heel')].z]

        left_leg = self.get_dist(left_heel, left_hip)
        right_leg = self.get_dist(right_heel, right_hip)

        return max(left_leg, right_leg)

    def save_data(self, reach_dist, leg_length):
        data_path = f'{self._ybt_data_out_folder}{self._class_name}_data.csv'

        if not os.path.exists(self._ybt_data_out_folder):
            os.makedirs(self._ybt_data_out_folder)

        if not os.path.exists(data_path):
            with open(data_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['文件名', '伸出距离', '下肢长度'])

        with open(data_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self._file_name] + [reach_dist, leg_length])

    def check_hand(self, landmarks):
        left_hip = [landmarks[self._landmark_names.index('left_hip')].x,
                    landmarks[self._landmark_names.index('left_hip')].y,
                    landmarks[self._landmark_names.index('left_hip')].z]

        left_wrist = [landmarks[self._landmark_names.index('left_wrist')].x,
                      landmarks[self._landmark_names.index('left_wrist')].y,
                      landmarks[self._landmark_names.index('left_wrist')].z]

        right_hip = [landmarks[self._landmark_names.index('right_hip')].x,
                     landmarks[self._landmark_names.index('right_hip')].y,
                     landmarks[self._landmark_names.index('right_hip')].z]

        right_wrist = [landmarks[self._landmark_names.index('right_wrist')].x,
                       landmarks[self._landmark_names.index('right_wrist')].y,
                       landmarks[self._landmark_names.index('right_wrist')].z]

        left_shoulder = [landmarks[self._landmark_names.index('left_shoulder')].x,
                         landmarks[self._landmark_names.index('left_shoulder')].y,
                         landmarks[self._landmark_names.index('left_shoulder')].z]

        right_shoulder = [landmarks[self._landmark_names.index('right_shoulder')].x,
                          landmarks[self._landmark_names.index('right_shoulder')].y,
                          landmarks[self._landmark_names.index('right_shoulder')].z]

        l_hand_hip = self.get_dist(left_hip, left_wrist)
        r_hand_hip = self.get_dist(right_hip, right_wrist)
        shoulder_dist = self.get_dist(left_shoulder, right_shoulder)

        # 判断左手是否叉腰
        if l_hand_hip <= shoulder_dist:
            l_hand_state = True
        else:
            l_hand_state = False

        # 判断右手是否叉腰
        if r_hand_hip <= shoulder_dist:
            r_hand_state = True
        else:
            r_hand_state = False

        return l_hand_state, r_hand_state

    def ybt_data_visulizer(self,
                           frame,
                           l_hand_state, r_hand_state,
                           max_reach,
                           landmarks):
        """
        对输出画面进行YBT数据渲染，在左右髋关节显示叉腰状态，在左下角显示最大伸出距离
        :param frame: 当前帧图像
        :param l_hand_state: 左手是否叉腰
        :param r_hand_state: 右手是否叉腰
        :param max_reach: 本次动作阶段的最大伸出距离
        :param landmarks: 渲染在关节所需要的相对坐标
        """

        output_img = Image.fromarray(frame)
        output_width = output_img.size[0]
        output_height = output_img.size[1]

        # 渲染叉腰状态
        # 根据状态显示颜色
        if l_hand_state:
            l_color = 'green'
        else:
            l_color = 'red'

        if r_hand_state:
            r_color = 'green'
        else:
            r_color = 'red'

        left_hip = [landmarks[self._landmark_names.index('left_hip')].x * output_width,
                    landmarks[self._landmark_names.index('left_hip')].y * output_height]

        right_hip = [landmarks[self._landmark_names.index('right_hip')].x * output_width,
                     landmarks[self._landmark_names.index('right_hip')].y * output_height]

        output_img_draw = ImageDraw.Draw(output_img)

        # 绘制左手叉腰状态
        output_img_draw.ellipse((left_hip[0]-30,left_hip[1]-30,left_hip[0]+30,left_hip[1]+30),
                                fill=l_color,
                                outline=None)

        # 绘制右手叉腰状态
        output_img_draw.ellipse((right_hip[0]-30,right_hip[1]-30,right_hip[0]+30,right_hip[1]+30),
                                fill=r_color,
                                outline=None)

        # 在左下角绘制最大伸出距离
        # 创建字体对象
        font_size = int(output_height * 0.1)
        font = ImageFont.truetype("Roboto-Regular.ttf", size=font_size)
        # 渲染文字
        text = str(max_reach)
        text_width, text_height = output_img_draw.textsize(text, font)
        output_img_draw.text((0, output_height - text_height), text, (255, 255, 255), font)

        return output_img

    def score(self):
        pass
