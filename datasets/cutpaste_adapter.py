"""
CutPaste适配器，使其兼容medsyn.tasks的接口
"""
import numpy as np
from PIL import Image
import random

class CutPasteWrapper:
    """
    将CutPaste包装成与medsyn.tasks相同的接口
    原接口：task(image) -> (augmented_image, mask)
    新接口：CutPaste(image) -> (original, augmented) 或 (original, normal_aug, scar_aug)
    """
    
    def __init__(self, cutpaste_type='3way', **cutpaste_kwargs):
        """
        参数:
            cutpaste_type: 'normal', 'scar', '3way'
            cutpaste_kwargs: 传递给CutPaste的参数
        """
        import sys
        sys.path.append('.')
        from cutpaste import CutPasteNormal, CutPasteScar, CutPaste3Way
        
        self.cutpaste_type = cutpaste_type
        
        if cutpaste_type == 'normal':
            self.cutpaste = CutPasteNormal(**cutpaste_kwargs)
        elif cutpaste_type == 'scar':
            self.cutpaste = CutPasteScar(**cutpaste_kwargs)
        elif cutpaste_type == '3way':
            self.cutpaste = CutPaste3Way(**cutpaste_kwargs)
        else:
            raise ValueError(f"Unknown cutpaste_type: {cutpaste_type}")
    
    def __call__(self, image):
        """
        输入: numpy array图像 (H, W) 或 (H, W, 1)
        输出: (augmented_image, mask)
        """
        # 将numpy数组转换为PIL图像
        if len(image.shape) == 2:  # (H, W)
            pil_image = Image.fromarray(image).convert('L').convert('RGB')
        elif len(image.shape) == 3 and image.shape[2] == 1:  # (H, W, 1)
            pil_image = Image.fromarray(image[:,:,0]).convert('L').convert('RGB')
        else:
            pil_image = Image.fromarray(image).convert('RGB')
        
        # 应用CutPaste增强
        if self.cutpaste_type == '3way':
            original, normal_aug, scar_aug = self.cutpaste(pil_image)
            
            # 随机选择一种增强结果
            if random.random() < 0.5:
                augmented = normal_aug
            else:
                augmented = scar_aug
        else:
            original, augmented = self.cutpaste(pil_image)
        
        # 生成对应的mask（CutPaste区域的掩码）
        mask = self._generate_mask(original, augmented)
        
        # 将PIL图像转回numpy数组
        augmented_np = np.array(augmented)
        
        # 如果需要灰度图，转换为单通道
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            augmented_np = augmented_np.mean(axis=2).astype(np.uint8)
        
        return augmented_np, mask
    
    def _generate_mask(self, original, augmented):
        """
        生成一个简单的掩码，标记修改过的区域
        这是一个简化版本，实际应用中可能需要更精确的掩码
        """
        import numpy as np
        
        orig_np = np.array(original)
        aug_np = np.array(augmented)
        
        # 计算像素差异
        if len(orig_np.shape) == 3:
            diff = np.abs(orig_np.astype(float) - aug_np.astype(float)).mean(axis=2)
        else:
            diff = np.abs(orig_np.astype(float) - aug_np.astype(float))
        
        # 创建二值掩码
        mask = (diff > 10).astype(np.float32)
        
        return mask
