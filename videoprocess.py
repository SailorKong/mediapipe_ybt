# encoding utf-8
"""
 @Author: Sailor
 @FileName: videoprocess.py
 @DateTime: 2023.01.31 11:48
 @SoftWare: PyCharm
"""
from matplotlib import pyplot as plt
import cv2
import numpy as np
import tqdm
import os
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import poseembedding as pe                      # 姿态关键点编码模块
import poseclassifier as pc                     # 姿态分类器
import resultsmooth as rs                       # 分类结果平滑
import counter                                  # 动作计数器
import ybt                                      # YBT评分模块
import visualizer as vs                         # 可视化模块


def save_image(img, path, figsize=(10, 10)):
    """save output PIL image."""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.savefig(path)
    plt.close()


def video_process(video_path, flag, file_name):
    # 指定视频路径和输出名称
    # video_path = 'A_LA-sample.mp4'
    # class_name需要与你的训练样本的两个动作状态图像文件夹的名字中的一个
    # （或者是与fitness_poses_csvs_out中的一个csv文件的名字）保持一致，它后面将用于分类时的索引。
    # 具体是哪个动作文件夹的名字取决于你的运动是什么，
    # 例如：如果是深蹲，明显比较重要的判断计数动作是蹲下去；
    #      如果是引体向上，则判断计数的动作是向上拉到最高点的那个动作；
    #      如果是俯卧撑，则判断计数的动作是最低点的那个动作
    pose_name = flag

    class_name = f'{pose_name}_DOWN'  # class_name 设置成 class_name_down 用于计数器判断
    out_video_path = f'videos_out/{pose_name}/{video_path[-9:-4]}_sample_out.mp4'
    last_frame_path = f'videos_out/{pose_name}/{video_path[-9:-4]}_last_frame.jpg'
    ybt_data_folder_path = f'ybt_data_out/'

    if not os.path.exists(f'videos_out/{pose_name}'):
        os.makedirs(f'videos_out/{pose_name}')

    # Open the video.
    video_cap = cv2.VideoCapture(video_path)

    # Get some video parameters to generate output video with classification.
    # 提取视频参数用于生成输出视频
    video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 视频总帧数
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)  # 视频FPS
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize tracker, classifier and counter.
    # 初始化pose检测、分类、计数模块
    # Do that before every video as all of them have stated.

    # Folder with pose class CSVs. That should be the same folder you're using while
    # building classifier to output CSVs.
    pose_samples_folder = f'poses_csvs_out/{pose_name}_poses_csvs_out'

    # Initialize tracker.
    pose_tracker = mp_pose.Pose()

    # Initialize embedder.
    pose_embedder = pe.FullBodyPoseEmbedder()

    # Initialize classifier.
    # Check that you are using the same parameters as during bootstrapping.
    pose_classifier = pc.PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    # # Uncomment to validate target poses used by classifier and find outliers.
    # # 取消注释就可以通过分类器和清除异常值的步骤对目标姿势进行验证
    # outliers = pose_classifier.find_pose_sample_outliers()
    # print('Number of pose sample outliers (consider removing them): ', len(outliers))

    # Initialize EMA smoothing.
    # 初始化指数移动平均EMA的平滑模块
    pose_classification_filter = rs.EMADictSmoothing(
        window_size=10,
        alpha=0.2)

    # Initialize YBT helper
    # 初始化YBT测试模块
    ybt_helper = ybt.YbtHelper(ybt_data_out_folder=ybt_data_folder_path,
                               class_name = pose_name,
                               file_name = file_name)

    # Initialize counter.
    # 初始化计数器
    repetition_counter = counter.RepetitionCounter(
        class_name=class_name,
        enter_threshold=6,
        exit_threshold=3)

    # Initialize renderer.
    # 初始化可视化渲染
    pose_classification_visualizer = vs.PoseClassificationVisualizer(
        class_name=class_name,
        plot_x_max=video_n_frames,
        # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
        plot_y_max=10)

    # Run classification on a video.

    # Open output video.
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

    frame_idx = 0
    output_frame = None
    reach_dists = []
    leg_dists = []
    with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
        while True:
            # Get next frame of the video.
            success, input_frame = video_cap.read()
            if not success:
                break

            # Run pose tracker.
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            result = pose_tracker.process(image=input_frame)
            pose_landmarks = result.pose_landmarks
            world_landmarks = result.pose_world_landmarks.landmark
            landmarks = pose_landmarks.landmark

            # Draw pose prediction.
            output_frame = input_frame.copy()
            if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)

            if pose_landmarks is not None:
                # Get landmarks.
                frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]

                # 根据图片像素定位关键点坐标，并保存为np.array()
                pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                           for lmk in pose_landmarks.landmark], dtype=np.float32)
                assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                # 获取当前帧画面中的到达距离和腿长
                reach_dist = ybt_helper.get_reach(world_landmarks)
                leg_length = ybt_helper.get_leg(world_landmarks)

                # 将到达距离和腿长进行最大值筛选
                reach_dists.append(reach_dist)
                leg_dists.append(leg_length)

                if len(reach_dists) > 6:
                    reach_dists = sorted(reach_dists, reverse=True)[:6]
                if len(leg_dists) >6:
                    leg_dists = sorted(leg_dists, reverse=True)[:6]

                # 双手叉腰检查
                l_hand_state, r_hand_state = ybt_helper.check_hand(world_landmarks)

                # Classify the pose on the current frame.
                pose_classification = pose_classifier(pose_landmarks)

                # Smooth classification using EMA.
                pose_classification_filtered = pose_classification_filter(pose_classification)

                # Count repetitions.
                repetitions_count = repetition_counter(pose_classification_filtered)
            else:
                # No pose => no classification on current frame.
                pose_classification = None

                # Still add empty classification to the filter to maintaing correct
                # smoothing for future frames.
                pose_classification_filtered = pose_classification_filter(dict())
                pose_classification_filtered = None

                # Don't update the counter presuming that person is 'frozen'. Just
                # take the latest repetitions count.
                repetitions_count = repetition_counter.n_repeats

            # Draw classification plot and repetition counter.
            # 对输出画面进行计数渲染
            output_frame = pose_classification_visualizer(
                frame=output_frame,
                pose_classification=pose_classification,
                pose_classification_filtered=pose_classification_filtered,
                repetitions_count=repetitions_count)

            # 对输出画面进行YBT数据渲染，在左右髋关节显示叉腰状态，在左下角显示最大伸出距离
            output_frame = ybt_helper.ybt_data_visulizer(
                frame=np.asarray(output_frame),
                l_hand_state=l_hand_state,
                r_hand_state=r_hand_state,
                max_reach = reach_dists[0],
                landmarks=landmarks)
            # Save the output frame.
            out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

            frame_idx += 1
            pbar.update()

    # Close output video.
    out_video.release()

    # Release MediaPipe resources.
    pose_tracker.close()

    # 获取最远到达距离和下肢长度
    max_reach = reach_dists[0]
    max_leg = leg_dists[0]

    # 保存ybt data，该动作下的最大到达长度和下肢长度
    ybt_helper.save_data(max_reach, max_leg)

    # Save the last frame of the video.
    if output_frame is not None:
        save_image(output_frame, last_frame_path)
