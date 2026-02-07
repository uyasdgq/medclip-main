import os  # 导入操作系统接口模块，用于文件路径操作
import random  # 导入随机数生成模块，用于数据采样和增强
from enum import Enum  # 导入枚举类，用于定义常量
import PIL  # 导入Python图像处理库
import torch  # 导入PyTorch深度学习框架
from torchvision import transforms  # 导入PyTorch的图像变换模块
import json  # 导入JSON处理模块，用于读取元数据文件
from PIL import Image  # 导入PIL图像处理类
import numpy as np  # 导入数值计算库

# 导入medsyn库中的各种异常合成任务类
from medsyn.tasks import CutPastePatchBlender  # 补丁混合任务，从其他图像剪切补丁粘贴到目标图像
from medsyn.tasks import SmoothIntensityChangeTask  # 平滑亮度变化任务，模拟局部阴影或过曝
from medsyn.tasks import GaussIntensityChangeTask  # 高斯亮度变化任务，模拟质地不均产生的纹理异常
from medsyn.tasks import SinkDeformationTask  # 凹陷形变任务，模拟组织收缩
from medsyn.tasks import SourceDeformationTask  # 凸起形变任务，模拟组织膨胀
from medsyn.tasks import IdentityTask  # 身份变换任务，不进行任何修改作为对照
try:
    from .cutpaste_adapter import CutPasteWrapper  # 尝试导入CutPaste封装器
except ImportError:
    from datasets.cutpaste_adapter import CutPasteWrapper  # 如果相对导入失败，使用绝对导入
import sys  # 导入系统模块
sys.path.append('.')  # 将当前目录添加到Python路径，方便导入
from cutpaste import CutPaste3Way, CutPasteNormal, CutPasteScar  # 导入CutPaste相关类
from .cutpaste_adapter import CutPasteWrapper  # 再次导入CutPaste封装器
class TrainDataset(torch.utils.data.Dataset):
    """
    训练数据集类
    负责加载训练数据，应用各种异常合成增强方法，为模型提供训练样本
    这是MediCLIP项目的核心数据加载器，集成了多种自监督异常合成技术
    """

    def __init__(
        self,
        args,  # 配置参数对象，包含所有训练相关的设置
        source,  # 数据源目录路径
        preprocess,  # 图像预处理函数（如标准化、调整大小等）
        k_shot = -1,  # 少样本学习的样本数量，-1表示使用全部数据
        **kwargs,  # 其他额外参数
    ):

        super().__init__()  # 调用父类Dataset的初始化方法
        self.args = args  # 保存配置参数
        self.source = source  # 保存数据源路径
        self.k_shot = k_shot  # 保存少样本设置
        self.transform_img = preprocess  # 保存图像预处理函数
        self.data_to_iterate = self.get_image_data()  # 获取并保存图像元数据列表
        self.augs,self.augs_pro = self.load_anomaly_syn()  # 加载异常合成任务及其概率权重
        assert sum(self.augs_pro)==1.0  # 确保所有任务的概率权重之和为1.0

    def __getitem__(self, idx):  # 根据索引获取单个数据样本的方法，PyTorch Dataset的必需方法
        """
        获取指定索引的训练样本
        
        Args:
            idx: 样本索引值
            
        Returns:
            dict: 包含'image'和'mask'键的字典，分别为增强后的图像张量和异常掩码张量
        """
        info = self.data_to_iterate[idx]  # 从待迭代数据列表中获取当前索引的元数据信息（包含文件名等）
        image_path=os.path.join(self.source,'images',info['filename'])  # 拼接图像的完整文件路径
        image = self.read_image(image_path)  # 调用自定义方法读取图像并预处理为numpy数组格式
        
        # 从已加载的增强任务列表中随机选择一个任务进行数据增强
        choice_aug = np.random.choice(a=[aug for aug in self.augs],  # 可用的增强任务列表
                                         p = [pro for pro in self.augs_pro],  # 每个任务对应的概率权重
                                         size=(1,), replace=False)[0]  # 只选择一个任务，不放回抽样
        
        image, mask = choice_aug(image)  # 应用选中的增强任务，生成增强后的图像和对应的异常定位掩码
        
        # 将numpy数组转换为PIL图像对象，并确保为RGB模式（3通道）
        image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
        image = self.transform_img(image)  # 应用传入的预处理变换（如标准化、调整大小等）
        mask = torch.from_numpy(mask)  # 将numpy掩码数组转换为PyTorch张量
        
        return {  # 返回包含图像和掩码的字典格式数据
            "image": image,  # 增强后的图像张量
            "mask" : mask,   # 异常区域的二值掩码张量
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def read_image(self,path):  # 自定义图像读取方法
        """
        读取并预处理图像文件
        
        Args:
            path: 图像文件的完整路径
            
        Returns:
            numpy.ndarray: 预处理后的图像数组，数据类型为uint8
        """
        # 使用PIL打开图像文件，调整大小为配置指定的尺寸，使用双线性插值，并转换为灰度图
        image = PIL.Image.open(path).resize((self.args.image_size,self.args.image_size),
                                            PIL.Image.Resampling.BILINEAR).convert("L")
        image = np.array(image).astype(np.uint8)  # 将PIL图像转换为numpy数组，确保数据类型为8位无符号整数
        return image  # 返回处理后的图像数组

    def get_image_data(self):  # 获取图像元数据的方法
        """
        从JSON文件中读取训练图像的元数据信息
        
        Returns:
            list: 包含图像元数据的字典列表，每个字典包含filename等信息
        """
        data_to_iterate = []  # 初始化空列表用于存储元数据
        # 打开训练样本的JSON文件，该文件包含所有训练图像的信息
        with open(os.path.join(self.source,'samples',"train.json"), "r") as f_r:
            for line in f_r:  # 逐行读取JSON文件（每行一个JSON对象）
                meta = json.loads(line)  # 解析JSON字符串为Python字典
                data_to_iterate.append(meta)  # 将元数据添加到列表中
                
        # 如果设置了少样本学习（k_shot不为-1）
        if self.k_shot != -1:
            # 从所有数据中随机采样指定数量的样本
            data_to_iterate = random.sample(
                data_to_iterate, self.k_shot  # 采样k_shot个样本
            )
        return data_to_iterate  # 返回元数据列表

    def load_anomaly_syn(self):
        """
        根据配置文件加载各种异常合成任务
        """
        tasks = []
        task_probability = []

        for task_name in self.args.anomaly_tasks.keys():
            prob = self.args.anomaly_tasks[task_name]
            if prob <= 0:
                continue

            if task_name == 'CutpasteTask':
                task = CutPasteWrapper(
                    cutpaste_type='3way',
                    area_ratio=[0.02, 0.12], # 稍微减小范围提高精细度
                    aspect_ratio=0.3,
                    width=[2, 12],
                    height=[10, 20],
                    rotation=[-45, 45],
                    colorJitter=0.08
                )
            elif task_name == 'SynomalyTask':
                task = SynomalyWrapper(
                    anomaly_sigma=random.uniform(5, 10), # 增加sigma的随机性
                    anomaly_threshold=random.randint(140, 160),
                    anomaly_offset=0.4 # 降低偏移量使异常更微妙
                )
            elif task_name == 'SmoothIntensityTask':
                task = SmoothIntensityChangeTask(25.0)
            elif task_name == 'GaussIntensityChangeTask':
                task = GaussIntensityChangeTask()
            elif task_name == 'SinkTask':
                task = SinkDeformationTask()
            elif task_name == 'SourceTask':
                task = SourceDeformationTask()
            elif task_name == 'IdentityTask':
                task = IdentityTask()
            else:
                raise NotImplementedError(f"Task {task_name} not supported")

            tasks.append(task)
            task_probability.append(prob)
        
        # 归一化概率
        total_prob = sum(task_probability)
        task_probability = [p / total_prob for p in task_probability]
        
        return tasks, task_probability

class ChexpertTestDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        args,
        source,
        preprocess,
        **kwargs,
    ):
        super().__init__()
        self.args = args
        self.source = source

        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()


    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path = os.path.join(self.source,'images',info['filename'])
        image = PIL.Image.open(image_path).convert("RGB").resize((self.args.image_size,self.args.image_size),PIL.Image.Resampling.BILINEAR)
        mask = np.zeros((self.args.image_size,self.args.image_size)).astype(float)
        image = self.transform_img(image)
        mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask" : mask,
            "classname": info['clsname'],
            "is_anomaly": info['label'],
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source,'samples',"test.json"), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        return data_to_iterate


class BrainMRITestDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        args,
        source,
        preprocess,
        **kwargs,
    ):

        super().__init__()
        self.args = args
        self.source = source
        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()


    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path = os.path.join(self.source,'images',info['filename'])
        image = PIL.Image.open(image_path).convert("RGB").resize((self.args.image_size,self.args.image_size),PIL.Image.Resampling.BILINEAR)
        mask = np.zeros((self.args.image_size,self.args.image_size)).astype(float)
        image = self.transform_img(image)
        mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask" : mask,
            "classname": info['clsname'],
            "is_anomaly": info['label'],
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)


    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source,'samples',"test.json"), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        return data_to_iterate



class BusiTestDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        args,
        source,
        preprocess,
        **kwargs,
    ):

        super().__init__()
        self.args = args
        self.source = source
        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()


    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path = os.path.join(self.source,'images',info['filename'])
        image = PIL.Image.open(image_path).convert("RGB").resize((self.args.image_size,self.args.image_size),PIL.Image.Resampling.BILINEAR)

        if info.get("mask", None):
            mask = os.path.join(self.source,'images',info['mask'])
            mask = PIL.Image.open(mask).convert("L").resize((self.args.image_size,self.args.image_size),PIL.Image.Resampling.NEAREST)
            mask = np.array(mask).astype(float)/255.0
            mask [mask!=0.0] = 1.0
        else:
            mask = np.zeros((self.args.image_size,self.args.image_size)).astype(float)

        image = self.transform_img(image)
        mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask": mask,
            "classname": info['clsname'],
            "is_anomaly": info['label'],
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source,'samples',"test.json"), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        return data_to_iterate