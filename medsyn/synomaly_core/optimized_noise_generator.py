import numpy as np
import cv2
import torch
import random
import skimage.exposure

def generate_synomaly_noise_for_mediclip(image, anomaly_sigma=5, anomaly_threshold=120, anomaly_offset=0.3, anomaly_direction=1):
    """
    优化后的Synomaly异常生成函数
    主要改进：
    1. 降低默认异常强度，避免过明显
    2. 增加随机性，提高多样性
    3. 优化阈值设置
    """
    
    # Ensure image is float for processing
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = image.astype(np.float32)

    # Handle shape
    if len(img_float.shape) == 3:
        height, width = img_float.shape[:2]
        is_color = True
    else:
        height, width = img_float.shape
        is_color = False
        
    # 1. Generate Gaussian background noise with reduced randomness
    background_noise = np.random.randn(height, width) * 0.8  # 降低噪声强度

    # 2. Create blob shapes using Gaussian Blur + Thresholding
    # 使用更小的sigma，生成更精细的异常
    blur = cv2.GaussianBlur(
        background_noise,
        (0, 0),
        sigmaX=anomaly_sigma,
        sigmaY=anomaly_sigma,
        borderType=cv2.BORDER_DEFAULT
    )
    
    # Rescale to 0-255 for thresholding
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255))
    
    # 使用更低的阈值，增加异常密度
    _, shape_mask = cv2.threshold(stretch, anomaly_threshold, 1, cv2.THRESH_BINARY)
    
    # 3. Apply intensity shift with reduced magnitude
    augmented_image = img_float.copy()
    
    # Determine direction
    if isinstance(anomaly_direction, list):
        current_direction = random.choice(anomaly_direction)
    else:
        current_direction = anomaly_direction
        
    # 降低强度变化幅度
    intensity_shift = current_direction * (np.random.rand() * 0.15 + (anomaly_offset * 0.3))
    
    # Apply to mask
    mask_bool = shape_mask == 1
    
    if is_color:
        for c in range(3):
            augmented_image[:, :, c][mask_bool] += intensity_shift
    else:
        augmented_image[mask_bool] += intensity_shift
        
    # Clip to valid range
    augmented_image = np.clip(augmented_image, 0.0, 1.0)
    
    # Convert back to uint8 if original was uint8
    if image.dtype == np.uint8:
        augmented_image = (augmented_image * 255).astype(np.uint8)
        
    return augmented_image, shape_mask.astype(np.float32)
