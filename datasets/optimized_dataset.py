import os
import random
from enum import Enum
import PIL
import torch
from torchvision import transforms
import json
from PIL import Image
import numpy as np

from medsyn.tasks import CutPastePatchBlender,\
                        SmoothIntensityChangeTask,\
                        GaussIntensityChangeTask,\
                        SinkDeformationTask,\
                        SourceDeformationTask,\
                        IdentityTask
from .optimized_synomaly_adapter import OptimizedSynomalyWrapper
from .cutpaste_adapter import CutPasteWrapper

class OptimizedTrainDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        args,
        source,
        preprocess,
        k_shot = -1,
        **kwargs,
    ):

        super().__init__()
        self.args = args
        self.source = source
        self.k_shot = k_shot
        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()
        self.augs,self.augs_pro = self.load_optimized_anomaly_syn()
        assert sum(self.augs_pro)==1.0

    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path=os.path.join(self.source,'images',info['filename'])
        image = self.read_image(image_path)
        
        # 使用优化后的任务选择
        choice_aug = np.random.choice(a=[aug for aug in self.augs],
                                         p = [pro for pro in self.augs_pro],
                                         size=(1,), replace=False)[0]
        image, mask = choice_aug(image)
        image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
        image = self.transform_img(image)
        mask = torch.from_numpy(mask)
        return {
            "image": image,
            "mask" : mask,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def read_image(self,path):
        image = PIL.Image.open(path).resize((self.args.image_size,self.args.image_size),
                                            PIL.Image.Resampling.BILINEAR).convert("L")
        image = np.array(image).astype(np.uint8)
        return image

    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source,'samples',"train.json"), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        if self.k_shot != -1:
            data_to_iterate = random.sample(
                data_to_iterate, self.k_shot
            )
        return data_to_iterate

    def load_optimized_anomaly_syn(self):
        """优化后的异常合成任务加载"""
        tasks = []
        task_probability = []

        for task_name in self.args.anomaly_tasks.keys():
            if task_name == 'CutpasteTask':
                # 使用更保守的CutPaste参数
                task = CutPasteWrapper(
                    cutpaste_type='3way',
                    area_ratio=[0.01, 0.1],  # 降低面积
                    aspect_ratio=0.3,
                    width=[2, 12],  # 降低宽度
                    height=[8, 20],  # 降低高度
                    rotation=[-30, 30],  # 降低旋转角度
                    colorJitter=0.05,  # 降低颜色抖动
                    transform=None
                )

            elif task_name == 'SynomalyTask':
                # 使用优化后的Synomaly
                task = OptimizedSynomalyWrapper(
                    anomaly_sigma=5,      # 降低默认值
                    anomaly_threshold=120, # 降低阈值
                    anomaly_offset=0.3     # 降低偏移
                )

            elif task_name == 'SmoothIntensityTask':
                task = SmoothIntensityChangeTask(20.0)  # 降低强度

            elif task_name == 'GaussIntensityChangeTask':
                task = GaussIntensityChangeTask()

            elif task_name == 'SinkTask':
                task = SinkDeformationTask()
            elif task_name == 'SourceTask':
                task = SourceDeformationTask()
            elif task_name == 'IdentityTask':
                task = IdentityTask()
            else:
                raise NotImplementedError("task must in [CutpasteTask, "
                                          "SmoothIntensityTask, "
                                          "GaussIntensityChangeTask,"
                                          "SinkTask, SourceTask, IdentityTask]")

            tasks.append(task)
            task_probability.append(self.args.anomaly_tasks[task_name])
        return tasks, task_probability
