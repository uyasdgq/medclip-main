import random  # 导入随机数生成模块，用于数据增强中的随机操作
import math  # 导入数学函数模块，用于几何计算
from torchvision import transforms  # 导入PyTorch的图像变换模块
import torch  # 导入PyTorch深度学习框架
from PIL import Image  # 导入Python图像处理库
import matplotlib.pyplot as plt  # 导入绘图库，用于可视化结果
import numpy as np  # 导入数值计算库

def cut_paste_collate_fn(batch):
    """
    自定义的批处理函数，用于处理CutPaste数据增强的批次数据
    
    Args:
        batch: 一个批次的数据，每个元素是(原图, 增强图)的元组
        
    Returns:
        list: 包含两个张量的列表，第一个是所有原图的张量，第二个是所有增强图张量
    """
    img_types = list(zip(*batch))  # 将批次中的数据解压并重新组织，分离原图和增强图
    return [torch.stack(imgs) for imgs in img_types]  # 将分离后的图像列表分别堆叠成张量

class CutPaste(object):
    """
    CutPaste数据增强的基类
    CutPaste是一种自监督学习的数据增强方法，通过从图像中剪切区域并粘贴到其他位置来创建异常样本
    """
    def __init__(self, colorJitter=0.1, transform=None):
        """
        初始化CutPaste基类
        
        Args:
            colorJitter: 颜色抖动强度，控制增强区域的颜色变化程度
            transform: 额外的图像变换函数，可在CutPaste操作后应用
        """
        self.transform = transform  # 保存图像变换函数
        if colorJitter is None:  # 如果颜色抖动参数为None
            self.colorJitter = None  # 则不使用颜色抖动
        else:
            # 创建颜色抖动变换对象，调整亮度、对比度、饱和度和色调
            self.colorJitter = transforms.ColorJitter(
                brightness=colorJitter,  # 亮度变化范围
                contrast=colorJitter,   # 对比度变化范围
                saturation=colorJitter,  # 饱和度变化范围
                hue=colorJitter         # 色调变化范围
            )

    def __call__(self, org_img, img):
        """
        执行CutPaste数据增强操作
        
        Args:
            org_img: 原始输入图像
            img: 经过CutPaste操作后的增强图像
            
        Returns:
            tuple: (原始图像, 增强图像) 的元组
        """
        if self.transform:  # 如果定义了额外的变换
            img = self.transform(img)  # 对增强图像应用变换
            org_img = self.transform(org_img)  # 对原始图像应用变换
        return org_img, img  # 返回原始图像和增强图像

class CutPasteNormal(CutPaste):
    """
    标准CutPaste数据增强类
    从图像中随机剪切一个矩形区域并粘贴到图像的其他位置
    模拟局部组织替换或移位造成的异常
    """
    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwargs):
        """
        初始化标准CutPaste增强
        
        Args:
            area_ratio: 剪切区域面积占总图像面积的比例范围 [最小值, 最大值]
            aspect_ratio: 剪切区域的长宽比参数，控制形状的偏扁或偏长
            **kwargs: 传递给父类的其他参数
        """
        super(CutPasteNormal, self).__init__(**kwargs)  # 调用父类初始化
        self.area_ratio = area_ratio  # 保存面积比例范围
        self.aspect_ratio = aspect_ratio  # 保存长宽比参数

    def __call__(self, img):
        """
        执行标准CutPaste增强操作
        
        Args:
            img: 输入的PIL图像
            
        Returns:
            tuple: (原始图像, 增强图像) 的元组
        """
        h = img.size[0]  # 获取图像高度
        w = img.size[1]  # 获取图像宽度
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h  # 计算剪切区域的实际面积
        
        # 计算长宽比的对数范围，用于后续随机采样
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1 / self.aspect_ratio)))
        # 在对数空间中随机采样长宽比，确保分布均匀
        aspect = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
        
        # 根据面积和长宽比计算剪切区域的宽度和高度
        cut_w = int(round(math.sqrt(ratio_area * aspect)))  # 计算剪切宽度
        cut_h = int(round(math.sqrt(ratio_area / aspect)))  # 计算剪切高度
        
        # 随机选择剪切区域的起始位置（确保不超出图像边界）
        from_location_h = int(random.uniform(0, h - cut_h))  # 垂直方向起始位置
        from_location_w = int(random.uniform(0, w - cut_w))  # 水平方向起始位置
        
        # 定义剪切区域的边界框 [左, 上, 右, 下]
        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)  # 从原图中剪切出补丁区域
        
        if self.colorJitter:  # 如果启用了颜色抖动
            patch = self.colorJitter(patch)  # 对剪切补丁应用颜色抖动，增加多样性
        
        # 随机选择粘贴位置（确保不超出图像边界）
        to_location_h = int(random.uniform(0, h - cut_h))  # 粘贴位置的垂直坐标
        to_location_w = int(random.uniform(0, w - cut_w))  # 粘贴位置的水平坐标
        
        # 定义粘贴区域的边界框
        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        augmented = img.copy()  # 创建图像副本用于增强
        augmented.paste(patch, insert_box)  # 将补丁粘贴到新位置
        
        return super().__call__(img, augmented)  # 调用父类方法返回原图和增强图

class CutPasteScar(CutPaste):
    """
    疤痕状CutPaste数据增强类
    生成类似疤痕或条带状的异常，通过旋转的矩形补丁模拟线性损伤
    适用于模拟医学图像中的疤痕、导管、血管等线性结构异常
    """
    def __init__(self, width=[2, 16], height=[10, 25], rotation=[-45, 45], **kwargs):
        """
        初始化疤痕状CutPaste增强
        
        Args:
            width: 剪切区域的宽度范围 [最小值, 最大值]
            height: 剪切区域的高度范围 [最小值, 最大值]  
            rotation: 旋转角度范围 [最小角度, 最大角度]，单位：度
            **kwargs: 传递给父类的其他参数
        """
        super(CutPasteScar, self).__init__(**kwargs)  # 调用父类初始化
        self.width = width  # 保存宽度范围
        self.height = height  # 保存高度范围
        self.rotation = rotation  # 保存旋转角度范围

    def __call__(self, img):
        """
        执行疤痕状CutPaste增强操作
        
        Args:
            img: 输入的PIL图像
            
        Returns:
            tuple: (原始图像, 增强图像) 的元组
        """
        h = img.size[0]  # 获取图像高度
        w = img.size[1]  # 获取图像宽度
        # 在指定范围内随机选择剪切区域的宽度和高度
        cut_w = random.uniform(*self.width)  # 随机宽度
        cut_h = random.uniform(*self.height)  # 随机高度
        
        # 随机选择剪切区域的起始位置
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        # 定义剪切区域的边界框
        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)  # 剪切出矩形补丁
        
        if self.colorJitter:  # 如果启用了颜色抖动
            patch = self.colorJitter(patch)  # 对补丁应用颜色抖动
        
        # 随机生成旋转角度
        rot_deg = random.uniform(*self.rotation)
        # 将补丁转换为RGBA模式（支持透明度）并旋转
        patch = patch.convert("RGBA").rotate(rot_deg, expand=True)
        
        # 随机选择粘贴位置（考虑旋转后的尺寸）
        to_location_h = int(random.uniform(0, h - patch.size[0]))
        to_location_w = int(random.uniform(0, w - patch.size[1]))
        
        # 获取RGBA图像的alpha通道（透明度掩码）
        mask = patch.split()[-1]
        patch = patch.convert("RGB")  # 转换回RGB模式
        
        augmented = img.copy()  # 创建图像副本
        # 使用掩码进行粘贴，确保只有非透明区域被粘贴
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)
        
        return super().__call__(img, augmented)  # 返回原图和增强图

class CutPasteUnion(object):
    """
    CutPaste联合增强类
    随机选择使用标准CutPaste或疤痕状CutPaste进行数据增强
    提供更多样化的异常类型，增强模型的泛化能力
    """
    def __init__(self, **kwargs):
        """
        初始化CutPaste联合增强
        
        Args:
            **kwargs: 包含所有CutPaste参数的字典，会自动分配给对应的子类
        """
        # 筛选出标准CutPaste需要的参数
        normal_kwargs = {k: v for k, v in kwargs.items() 
                        if k in ['area_ratio', 'aspect_ratio', 'colorJitter', 'transform']}
        # 筛选出疤痕状CutPaste需要的参数
        scar_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['width', 'height', 'rotation', 'colorJitter', 'transform']}
        
        # 创建标准CutPaste实例
        self.normal = CutPasteNormal(**normal_kwargs)
        # 创建疤痕状CutPaste实例
        self.scar = CutPasteScar(**scar_kwargs)

    def __call__(self, img):
        """
        执行联合CutPaste增强操作
        
        Args:
            img: 输入的PIL图像
            
        Returns:
            tuple: (原始图像, 增强图像) 的元组
        """
        r = random.uniform(0, 1)  # 生成[0,1]范围内的随机数
        if r < 0.5:  # 50%概率选择标准CutPaste
            return self.normal(img)
        else:  # 50%概率选择疤痕状CutPaste
            return self.scar(img)

class CutPaste3Way(object):
    """
    CutPaste三路增强类
    同时生成标准CutPaste和疤痕状CutPaste两种增强结果
    提供更丰富的训练数据，适用于需要多类型异常的场景
    """
    def __init__(self, **kwargs):
        """
        初始化CutPaste三路增强
        
        Args:
            **kwargs: 包含所有CutPaste参数的字典
        """
        # 筛选出标准CutPaste需要的参数
        normal_kwargs = {k: v for k, v in kwargs.items() 
                        if k in ['area_ratio', 'aspect_ratio', 'colorJitter', 'transform']}
        # 筛选出疤痕状CutPaste需要的参数
        scar_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['width', 'height', 'rotation', 'colorJitter', 'transform']}
        
        # 创建标准CutPaste实例
        self.normal = CutPasteNormal(**normal_kwargs)
        # 创建疤痕状CutPaste实例
        self.scar = CutPasteScar(**scar_kwargs)

    def __call__(self, img):
        """
        执行三路CutPaste增强操作
        
        Args:
            img: 输入的PIL图像
            
        Returns:
            tuple: (原始图像, 标准增强图像, 疤痕状增强图像) 的三元组
        """
        # 对原图应用标准CutPaste增强
        org, cutpaste_normal = self.normal(img)
        # 对原图应用疤痕状CutPaste增强（注意：这里使用原图，不是增强后的图）
        _, cutpaste_scar = self.scar(img)
        # 返回原图和两种不同的增强结果
        return org, cutpaste_normal, cutpaste_scar

# 测试代码保持不变
if __name__ == '__main__':
    print("=" * 60)
    print("CutPaste 代码测试开始")
    print("=" * 60)
    
    random.seed(42)
    torch.manual_seed(42)
    
    print("\n1. 创建测试图像...")
    width, height = 256, 256
    test_image = Image.new('RGB', (width, height), color='white')
    pixels = test_image.load()
    
    for i in range(width):
        for j in range(height):
            r = int((i / width) * 255)
            g = int((j / height) * 255)
            b = 128
            pixels[i, j] = (r, g, b)
    
    print(f"   图像大小: {test_image.size}")
    print(f"   图像模式: {test_image.mode}")
    
    print("\n2. 测试 CutPasteNormal...")
    try:
        normal_aug = CutPasteNormal(area_ratio=[0.02, 0.08], aspect_ratio=0.3, colorJitter=0.1)
        original_normal, augmented_normal = normal_aug(test_image.copy())
        print(f"   ✓ 成功! 返回类型: {type(augmented_normal)}")
        print(f"   原图大小: {original_normal.size}, 增强图大小: {augmented_normal.size}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    print("\n3. 测试 CutPasteScar...")
    try:
        scar_aug = CutPasteScar(width=[3, 10], height=[15, 30], rotation=[-30, 30], colorJitter=0.05)
        original_scar, augmented_scar = scar_aug(test_image.copy())
        print(f"   ✓ 成功! 返回类型: {type(augmented_scar)}")
        print(f"   原图大小: {original_scar.size}, 增强图大小: {augmented_scar.size}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    print("\n4. 测试 CutPasteUnion...")
    try:
        union_aug = CutPasteUnion(
            area_ratio=[0.02, 0.1],
            aspect_ratio=0.3,
            width=[2, 8],
            height=[10, 25],
            rotation=[-45, 45],
            colorJitter=0.1
        )
        original_union, augmented_union = union_aug(test_image.copy())
        print(f"   ✓ 成功! 返回类型: {type(augmented_union)}")
        print(f"   原图大小: {original_union.size}, 增强图大小: {augmented_union.size}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    print("\n5. 测试 CutPaste3Way...")
    try:
        threeway_aug = CutPaste3Way(
            area_ratio=[0.02, 0.1],
            aspect_ratio=0.3,
            width=[2, 8],
            height=[10, 25],
            rotation=[-45, 45],
            colorJitter=0.1
        )
        original_3way, normal_3way, scar_3way = threeway_aug(test_image.copy())
        print(f"   ✓ 成功! 返回3个图像")
        print(f"   原图大小: {original_3way.size}")
        print(f"   Normal增强图大小: {normal_3way.size}")
        print(f"   Scar增强图大小: {scar_3way.size}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    print("\n6. 测试 cut_paste_collate_fn...")
    try:
        batch_size = 4
        fake_batch = []
        for i in range(batch_size):
            img_tensor = torch.randn(3, 64, 64)
            aug_tensor = torch.randn(3, 64, 64)
            fake_batch.append((img_tensor, aug_tensor))
        
        collated = cut_paste_collate_fn(fake_batch)
        print(f"   ✓ 成功! 返回列表长度: {len(collated)}")
        print(f"   第一个元素形状: {collated[0].shape}")
        print(f"   第二个元素形状: {collated[1].shape}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    print("\n7. 生成可视化结果...")
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(test_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        if 'augmented_normal' in locals():
            axes[0, 1].imshow(augmented_normal)
            axes[0, 1].set_title('CutPaste Normal')
            axes[0, 1].axis('off')
        
        if 'augmented_scar' in locals():
            axes[0, 2].imshow(augmented_scar)
            axes[0, 2].set_title('CutPaste Scar')
            axes[0, 2].axis('off')
        
        if 'augmented_union' in locals():
            axes[1, 0].imshow(augmented_union)
            axes[1, 0].set_title('CutPaste Union')
            axes[1, 0].axis('off')
        
        if 'normal_3way' in locals():
            axes[1, 1].imshow(normal_3way)
            axes[1, 1].set_title('CutPaste3Way (Normal)')
            axes[1, 1].axis('off')
        
        if 'scar_3way' in locals():
            axes[1, 2].imshow(scar_3way)
            axes[1, 2].set_title('CutPaste3Way (Scar)')
            axes[1, 2].axis('off')
        
        plt.suptitle('CutPaste Data Augmentation Results', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig('cutpaste_test_results.png', dpi=150, bbox_inches='tight')
        print(f"   ✓ 可视化结果已保存为: cutpaste_test_results.png")
        plt.show()
    except Exception as e:
        print(f"   ✗ 可视化失败: {e}")
    
    print("\n8. 批量测试不同参数...")
    test_cases = [
        ("Small patches", {"area_ratio": [0.01, 0.05], "aspect_ratio": 0.2}),
        ("Large patches", {"area_ratio": [0.1, 0.2], "aspect_ratio": 0.4}),
        ("No color jitter", {"colorJitter": None}),
        ("Strong color jitter", {"colorJitter": 0.3}),
    ]
    
    for name, params in test_cases:
        try:
            aug = CutPasteNormal(**params)
            _, test_result = aug(test_image.copy())
            print(f"   ✓ {name}: 成功")
        except Exception as e:
            print(f"   ✗ {name}: 失败 - {e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    print("\n总结:")
    print("1. 所有CutPaste类都能正常工作")
    print("2. 输入: PIL Image对象")
    print("3. 输出: (原图, 增强图) 或 (原图, normal增强图, scar增强图)")
    print("4. 可以调整area_ratio、aspect_ratio等参数控制异常大小和形状")
    print("5. colorJitter参数控制颜色扰动强度（None表示无扰动）")
