import numpy as np
import torch
from medsyn.synomaly_core.optimized_noise_generator import generate_synomaly_noise_for_mediclip

class OptimizedSynomalyWrapper:
    def __init__(self, anomaly_sigma=3, anomaly_threshold=175, anomaly_offset=0.5, **kwargs):
        """
        优化后的Synomaly封装器
        已调整为对齐论文参数 (Paper-Aligned Settings)
        """
        self.anomaly_sigma = anomaly_sigma  # Paper: 3
        self.anomaly_threshold = anomaly_threshold  # Paper: 175
        self.anomaly_offset = anomaly_offset  # Paper: 0.5
        # 添加参数随机化 (围绕论文参数微调)
        self.sigma_range = [2, 5]  # Paper sigma=3
        self.threshold_range = [150, 200]  # Paper threshold=175

    def __call__(self, image):
        """
        Args:
            image: Numpy array (H, W) or (H, W, C)
        Returns:
            aug_image: Augmented image
            mask: Binary mask (H, W)
        """
        # 添加参数随机化，增加多样性
        current_sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        current_threshold = np.random.uniform(self.threshold_range[0], self.threshold_range[1])
        
        aug_image, mask = generate_synomaly_noise_for_mediclip(
            image,
            anomaly_sigma=current_sigma,
            anomaly_threshold=current_threshold,
            anomaly_offset=self.anomaly_offset
        )
        
        return aug_image, mask
