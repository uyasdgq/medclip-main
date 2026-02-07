import numpy as np  # 导入numpy库，用于生成和处理数组数据
import torch  # 导入torch库，用于深度学习相关的处理
from medsyn.synomaly_core.noise_generator import generate_synomaly_noise_for_mediclip  # 从项目中导入合成噪声生成的辅助函数

class SynomalyWrapper:  # 定义一个用于Synomaly合成任务的封装类
    def __init__(self, anomaly_sigma=7, anomaly_threshold=150, anomaly_offset=0.5, **kwargs):  # 初始化方法，设置默认参数
        """
        Wrapper for Synomaly task.
        """
        self.anomaly_sigma = anomaly_sigma  # 控制噪声块的大小，sigma越大，生成的块越大
        self.anomaly_threshold = anomaly_threshold  # 阈值，控制生成异常的密度，值越高，异常区域越少
        self.anomaly_offset = anomaly_offset  # 异常偏移量，控制异常区域相对于原图的亮度变化程度
        # **kwargs 用于接收其他可能传入的多余参数，防止初始化报错

    def __call__(self, image):  # 使类实例可以像函数一样被调用
        """
        Args:
            image: Numpy array (H, W) or (H, W, C)  # 输入图像，可以是二维灰度图或三维彩色图
        Returns:
            aug_image: Augmented image  # 返回合成异常后的图像
            mask: Binary mask (H, W)  # 返回异常区域的二值掩码
        """
        aug_image, mask = generate_synomaly_noise_for_mediclip(  # 调用核心生成函数进行图像增强
            image,   # 原始图像
            anomaly_sigma=self.anomaly_sigma,  # 传入块大小参数
            anomaly_threshold=self.anomaly_threshold,  # 传入密度控制参数
            anomaly_offset=self.anomaly_offset  # 传入亮度偏移参数
        )
        
        return aug_image, mask  # 返回增强后的图像和对应的掩码

