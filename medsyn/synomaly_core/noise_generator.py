import numpy as np  # 导入常用的科学计算库numpy
import cv2  # 导入OpenCV库，用于图像处理操作，如高斯模糊
import torch  # 导入PyTorch库
import random  # 导入随机数生成模块
import skimage.exposure  # 导入scikit-image的曝光调整模块，用于亮度缩放

def generate_synomaly_noise_for_mediclip(image, anomaly_sigma=7, anomaly_threshold=150, anomaly_offset=0.5, anomaly_direction=1):  # 定义生成异常噪声的核心函数
    """
    Generate Synomaly-style anomalies for MediCLIP.
    Adapted from Synomaly's 'generate_synomaly_noise'.
    
    Args:
        image: Input image (numpy array, [H, W] or [H, W, C]), range [0, 255] or [0, 1].
        anomaly_sigma: Controls the size of the blobs (larger = bigger anomalies).
        anomaly_threshold: Controls density (larger = fewer anomalies).
        anomaly_offset: Intensity shift amount.
        anomaly_direction: 1 for hyper-intense, -1 for hypo-intense.
        
    Returns:
        augmented_image: Image with anomalies added.
        mask: Binary mask of where anomalies were added.
    """
    
    # Ensure image is float for processing
    if image.dtype == np.uint8:  # 检查输入图像是否为uint8类型（0-255）
        img_float = image.astype(np.float32) / 255.0  # 如果是，转换为float32并归一化到[0, 1]区间
    else:
        img_float = image.astype(np.float32)  # 已经是float类型，则仅确保数据类型统一

    # Handle shape
    if len(img_float.shape) == 3:  # 检查是否为彩色图像（含有通道维度）
        height, width = img_float.shape[:2]  # 获取高度和宽度
        is_color = True  # 标记为彩色图像
    else:
        height, width = img_float.shape  # 如果是灰度图，直接获取高度和宽度
        is_color = False  # 标记为灰度图像
        
    # 1. Generate Gaussian background noise
    background_noise = np.random.randn(height, width)  # 生成与图像尺寸一致的随机正态分布噪声背景

    # 2. Create blob shapes using Gaussian Blur + Thresholding
    # This simulates "diffuse" organic shapes like tumors or edema
    blur = cv2.GaussianBlur(  # 对随机噪声进行高斯模糊
        background_noise,   # 输入随机噪声
        (0, 0),   # 自动计算核大小
        sigmaX=anomaly_sigma,   # X方向的偏差值，控制生成的形状块大小
        sigmaY=anomaly_sigma,   # Y方向的偏差值，控制生成的形状块大小
        borderType=cv2.BORDER_DEFAULT  # 采用默认边界扩展方式
    )
    
    # Rescale to 0-255 for thresholding
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255))  # 将平滑后的噪声映射到0-255范围，方便后续二值化
    
    # Threshold to get binary mask of "blobs"
    # Note: cv2.threshold returns (retval, dst)
    _, shape_mask = cv2.threshold(stretch, anomaly_threshold, 1, cv2.THRESH_BINARY)  # 通过设定阈值，将高于阈值的区域设为1，低于的设为0，以此生成随机且平滑的斑块掩码
    
    # 3. Apply intensity shift to the image in masked regions
    augmented_image = img_float.copy()
    
    # Create soft edges (feathering) for the mask to avoid sharp signal transitions
    soft_mask = cv2.GaussianBlur(shape_mask.astype(np.float32), (0, 0), sigmaX=2.0)
    
    # 4. Level 2 Optimization: Heterogeneous Texture
    # Generate internal variation (multi-frequency noise) to simulate necrosis or calcification
    internal_texture = np.random.randn(height, width)
    internal_texture = cv2.GaussianBlur(internal_texture, (0, 0), sigmaX=1.0) 
    # Normalize texture to roughly [-0.5, 0.5]
    internal_texture = (internal_texture - np.mean(internal_texture)) / (np.std(internal_texture) + 1e-5) * 0.2
    
    # Determine direction
    if isinstance(anomaly_direction, list):
        current_direction = random.choice(anomaly_direction)
    else:
        current_direction = anomaly_direction
        
    # Base intensity shift
    base_shift = current_direction * (np.random.rand() * 0.2 + (anomaly_offset * 0.5))
    
    # Combine base shift with texture variation
    # Final shift = base_shift * (1 + texture)
    total_shift_map = base_shift * (1.0 + internal_texture)
    
    # Apply to image using the soft mask
    if is_color:
        for c in range(3):
            augmented_image[:, :, c] += total_shift_map * soft_mask
    else:
        augmented_image += total_shift_map * soft_mask
        
    # Clip to valid range
    augmented_image = np.clip(augmented_image, 0.0, 1.0)
    
    # Convert back to uint8 if original was uint8
    if image.dtype == np.uint8:
        augmented_image = (augmented_image * 255).astype(np.uint8)
        
    return augmented_image, soft_mask.astype(np.float32)

