# encoding utf-8
"""
 @Author: Sailor
 @FileName: trainingsetprocess.py
 @DateTime: 2023.01.31 11:48
 @SoftWare: PyCharm
"""

import poseembedding as pe  # 姿态关键点归一化编码模块
import poseclassifier as pc  # 姿态分类器
import extracttrainingsetkeypoints as ek  # 提取训练集关键点特征
import csv
import os


# Required structure of the images_in_folder:
#
#   A_LA_images_in/
#     A_LA_up/
#       image_001.jpg
#       image_002.jpg
#       ...
#     A_LA_down/
#       image_001.jpg
#       image_002.jpg
#       ...
#     ...

def trainset_process(flag):
    """
    训练器
    如果存在训练后生成的CSV文件，则不运行
    """
    # 如果A_LA_poses_csvs_out文件夹下的A_LA_DOWN.csv和A_LA_UP.csv已经存在，则不用导入样本图片再训练了
    if flag == 'A_LA':
        pose_name = flag
        if os.path.isfile('poses_csvs_out/A_LA_poses_csvs_out/A_LA_UP.csv') and os.path.isfile(
                'poses_csvs_out/A_LA_poses_csvs_out/A_LA_DOWN.csv'):
            return
    elif flag == 'A_RA':
        pose_name = flag
        if os.path.isfile('poses_csvs_out/A_RA_poses_csvs_out/A_RA_UP.csv') and os.path.isfile(
                'poses_csvs_out/A_RA_poses_csvs_out/A_RA_DOWN.csv'):
            return
    elif flag == 'L_LL':
        pose_name = flag
        if os.path.isfile('poses_csvs_out/L_LL_poses_csvs_out/L_LL_UP.csv') and os.path.isfile(
                'poses_csvs_out/L_LL_poses_csvs_out/L_LL_DOWN.csv'):
            return
    elif flag == 'L_RL':
        pose_name = flag
        if os.path.isfile('poses_csvs_out/L_RL_poses_csvs_out/L_RL_UP.csv') and os.path.isfile(
                'poses_csvs_out/L_RL_poses_csvs_out/L_RL_DOWN.csv'):
            return
    elif flag == 'M_LM':
        pose_name = flag
        if os.path.isfile('poses_csvs_out/M_LM_poses_csvs_out/M_LM_UP.csv') and os.path.isfile(
                'poses_csvs_out/M_LM_poses_csvs_out/M_LM_DOWN.csv'):
            return
    elif flag == 'M_RM':
        pose_name = flag
        if os.path.isfile('poses_csvs_out/M_RM_poses_csvs_out/M_RM_UP.csv') and os.path.isfile(
                'poses_csvs_out/M_RM_poses_csvs_out/M_RM_DOWN.csv'):
            return


    # 指定样本图片的路径，训练集文件夹
    bootstrap_images_in_folder = f'images_in/{pose_name}_images_in'

    # Output folders for bootstrapped images and CSVs，训练集训练后输出文件夹
    bootstrap_images_out_folder = f'images_out/{pose_name}_images_out'
    bootstrap_csvs_out_folder = f'poses_csvs_out/{pose_name}_poses_csvs_out'

    # Initialize helper.
    bootstrap_helper = ek.BootstrapHelper(
        images_in_folder=bootstrap_images_in_folder,
        images_out_folder=bootstrap_images_out_folder,
        csvs_out_folder=bootstrap_csvs_out_folder,
    )

    # Check how many pose classes and images for them are available.
    # 打印要分类的姿势名称 以及对应的图片数量
    bootstrap_helper.print_images_in_statistics()

    # Bootstrap all images.
    # 对训练集中的图片进行模型姿态检测，得到landmarks坐标后转化为图片像素坐标并保存成CSV文件
    # Set limit to some small number for debug.该limit参数用于限制图片数量来debug
    bootstrap_helper.bootstrap(per_pose_class_limit=None)

    # Check how many images were bootstrapped.
    # 检查训练集中多少图片被顺利处理
    bootstrap_helper.print_images_out_statistics()

    # After initial bootstrapping images without detected poses were still saved in
    # 删除没有识别出pose的图像
    # the folders(but not in the CSVs) for debug purpose. Let's remove them.
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()

    # Please manually verify predictions and remove samples (images) that has wrong pose prediction.
    # Check as if you were asked to classify pose just from predicted landmarks. If you can't - remove it.
    # 人工再检查一遍预测图片，并进行必要的清理
    # Align CSVs and image folders once you are done.图片清理后再删除CSV文件中的对应数据

    # Align CSVs with filtered images.
    # 再次对其CSV文件与图片数据，并打印剩余图片个数
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()

    # ## Automatic filtration
    # ## 自动筛选
    #
    # Classify each sample against database of all other samples
    # and check if it gets in the same class as annotated after classification.
    #
    # There can be two reasons for the outliers:
    #
    #   * **Wrong pose prediction**:
    #       In this case remove such outliers.
    #
    #   * **Wrong classification**
    #       (i.e. pose is predicted correctly, and you agree with original pose class assigned to the sample):
    #       In this case sample is from the under-represented group (e.g. unusual angle or just very few samples).
    #       Add more similar samples and run bootstrapping from the very beginning.
    #
    # Even if you just removed some samples it makes sense to re-run automatic filtration one more time
    # as database of poses has changed.
    #
    # **Important!!** Check that you are using the same parameters when classifying whole videos later.

    # Find outliers.

    # Transforms pose landmarks into embedding.
    # 将生成的landmarks坐标归一化的embedder对象
    pose_embedder = pe.FullBodyPoseEmbedder()

    # Classifies give pose against database of poses.
    pose_classifier = pc.PoseClassifier(
        pose_samples_folder=bootstrap_csvs_out_folder,
        pose_embedder=pose_embedder,
        # KNN算法中的n，即纳入几个临近数据作为参考
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    # 找出异常值
    # 该函数内部使用了KNN算法对动作进行分类，在分类后找出异常值
    outliers = pose_classifier.find_pose_sample_outliers()
    print('Number of outliers: ', len(outliers))

    # Analyze outliers. 输出异常值的信息及图片
    bootstrap_helper.analyze_outliers(outliers)

    # Remove all outliers (if you don't want to manually pick).
    bootstrap_helper.remove_outliers(outliers)

    # Align CSVs with images after removing outliers.
    # 删除异常值后重新整理文件夹图片和CSV文件
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()
