from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union
import functools
import itertools
import numpy as np
import numpy.typing as npt
from scipy.ndimage import affine_transform,distance_transform_edt
from numpy.linalg import norm
import random
from scipy import ndimage
from scipy.ndimage import gaussian_filter

from .labelling import FlippedGaussianLabeller,AnomalyLabeller
from .task_shape import EitherDeformedHypershapePatchMaker,PerlinPatchMaker
from .utils import *

def cut_paste(sample: npt.NDArray[float],
              source_to_blend: npt.NDArray[float],
              anomaly_corner: npt.NDArray[int],
              anomaly_mask: npt.NDArray[bool],
              feather_sigma: float = 2.0) -> npt.NDArray[float]:  # Added feathering parameter

    # Create a soft mask using Gaussian blur to avoid sharp edges
    soft_mask = gaussian_filter(anomaly_mask.astype(float), sigma=feather_sigma)
    repeated_mask = np.broadcast_to(soft_mask, source_to_blend.shape)

    sample_slices = get_patch_image_slices(anomaly_corner, tuple(anomaly_mask.shape))

    aug_sample = sample.copy()
    # Use alpha blending instead of hard replacement
    aug_sample[sample_slices] = aug_sample[sample_slices] * (1 - repeated_mask) + \
                               source_to_blend * repeated_mask

    return aug_sample



class BaseTask(ABC):  # 定义自监督任务的基础抽象类
    def __init__(self,
                 sample_labeller: Optional[AnomalyLabeller] = None,  # 可选的样本标注器，用于生成真实掩码
                 **all_kwargs):

        self.sample_labeller = sample_labeller  # 保存标注器
        self.rng = np.random.default_rng()  # 初始化numpy随机数生成器

        self.min_anom_prop=0.3  # 异常区域最小占比（相对于图像尺寸）
        self.max_anom_prop=0.8  # 异常区域最大占比

        self.anomaly_shape_maker = EitherDeformedHypershapePatchMaker(nsa_sample_dimension)  # 默认形状生成器
        self.all_kwargs = all_kwargs  # 保存其他参数


    def apply(self,
              sample: npt.NDArray[float],  # 输入图像样本
              sample_mask=None,
              *args, **kwargs)\
            -> Tuple[npt.NDArray[float], npt.NDArray[float]]:  # 返回(增强图像, 标签)
        """
        Apply the self-supervised task to the single data sample.
        :param sample: Normal sample to be augmented
        :param sample_mask: Object mask of sample.
        :return: sample with task applied and label map.
        """

        aug_sample = sample.copy()  # 复制样本，避免原地修改

        sample_shape = np.array(sample.shape[1:])  # 获取图像的空间维度大小（如H, W）
        anomaly_mask = np.zeros(sample_shape, dtype=bool)  # 初始化全黑的异常掩码

        min_anom_prop = self.min_anom_prop  # 获取占比下限
        max_anom_prop = self.max_anom_prop  # 获取占比上限

        min_dim_lens = (min_anom_prop * sample_shape).round().astype(int)  # 计算最小补丁尺寸
        max_dim_lens = (max_anom_prop * sample_shape).round().astype(int)  # 计算最大补丁尺寸
        # print(min_dim_lens,max_dim_lens) [15,15],[205,205]

        dim_bounds = list(zip(min_dim_lens, max_dim_lens)) #[(15, 205), (15, 205)]  # 打包每个维度的尺寸限制

        # For random number of times
        if sample_mask is None:
            # Level 2 Optimization: Automatically detect brain tissue (foreground)
            # Brain MRI background is typically near zero. Thresholding at 5% of max helps.
            sample_mask = (np.mean(sample, axis=0) > (np.max(sample) * 0.05))

        for i in range(2):  # 尝试生成最多两个异常区域

            # Compute anomaly mask
            curr_anomaly_mask, intersect_fn = self.anomaly_shape_maker.get_patch_mask_and_intersect_fn(dim_bounds,
                                                                                                       sample_shape)
            # 调用形状生成器获取当前随机补丁的掩码和相交检测函数

            # Choose anomaly location
            anomaly_corner = self.find_valid_anomaly_location(curr_anomaly_mask, sample_mask, sample_shape)
            # 在图像中寻找一个合法的位置来放置该补丁

            # Apply self-supervised task
            aug_sample = self.augment_sample(aug_sample, sample_mask, anomaly_corner, curr_anomaly_mask, intersect_fn)
            # 调用核心增强逻辑（在子类中实现，如CutPaste或亮度变化）

            anomaly_mask[get_patch_slices(anomaly_corner, curr_anomaly_mask.shape)] |= curr_anomaly_mask
            # 更新整体异常掩码，记录补丁覆盖的位置

            # Randomly brake at end of loop, ensuring we get at least 1 anomaly
            if self.rng.random() > 0.5:  # 50%的概率停止生成，或者继续生成第二个
                break

        if self.sample_labeller is not None:  # 如果提供了标注器
            return aug_sample, self.sample_labeller(aug_sample, sample, anomaly_mask)  # 生成最终的标签（可能是概率分布或二值掩码）
        else:
            # If no labeller is provided, we are probably in a calibration process
            return aug_sample, np.expand_dims(anomaly_mask, 0)  # 默认返回补丁掩码本身


    def find_valid_anomaly_location(self,
                                    curr_anomaly_mask: npt.NDArray[bool],
                                    sample_mask: Optional[npt.NDArray[bool]],
                                    sample_shape: npt.NDArray[int]):

        curr_anomaly_shape = np.array(curr_anomaly_mask.shape)
        min_corner = np.zeros(len(sample_shape))
        max_corner = sample_shape - curr_anomaly_shape

        # - Apply anomaly at location
        while True:
            anomaly_corner = self.rng.integers(min_corner, max_corner, endpoint=True)

            # If the sample mask is None, any location within the bounds is valid
            if sample_mask is None:
                break
            # Otherwise, we need to check that the intersection of the anomaly mask and the sample mask is at least 50%
            target_patch_obj_mask = sample_mask[get_patch_slices(anomaly_corner, curr_anomaly_mask.shape)]
            if (np.sum(target_patch_obj_mask & curr_anomaly_mask) / np.sum(curr_anomaly_mask)) >= 0.5:
                break

        return anomaly_corner


    def __call__(self,
                 sample: npt.NDArray[float],
                 *args,
                 **kwargs)\
            -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """
        Apply the self-supervised task to the single data sample.
        :param sample: Normal sample to be augmented
        :param sample_mask: Object mask of sample.
        :param **kwargs:
            * *sample_path*: Path to source image
        :return: sample with task applied and label map.
        """
        if len(sample.shape)==2:
            sample = np.expand_dims(sample,axis=0)

        aug_sample, aug_mask = self.apply(sample, *args, **kwargs)

        if len(aug_sample.shape)==3 and aug_sample.shape[0]==1:
            aug_sample = aug_sample.squeeze(0)

        if len(aug_mask.shape)==3 and aug_mask.shape[0]==1:
            aug_mask = aug_mask.squeeze(0)

        return aug_sample,aug_mask.astype(float)


    @abstractmethod
    def augment_sample(self, sample: npt.NDArray[float], sample_mask: Optional[npt.NDArray[bool]],
                       anomaly_corner: npt.NDArray[int], anomaly_mask: npt.NDArray[bool],
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:
        """
        Apply self-supervised task to region at anomaly_corner covered by anomaly_mask
        :param sample: Sample to be augmented.
        :param sample_mask: Object mask of sample.
        :param anomaly_corner: Index of anomaly corner.
        :param anomaly_mask: Mask
        :param anomaly_intersect_fn: Function which, given a line's origin and direction, finds its intersection with
        the edge of the anomaly mask
        :return:
        """


class BasePatchBlendingTask(BaseTask):  # 补丁混合类任务的基类

    def __init__(self,
                 sample_labeller: Optional[AnomalyLabeller],  # 标注器
                 source_samples: list,  # 源图像列表，用于从中提取补丁
                 blend_images: Callable[[npt.NDArray[float], npt.NDArray[float], npt.NDArray[int], npt.NDArray[bool]],
                                        npt.NDArray[float]],  # 定义如何将补丁混合到目标图像的函数

                 **all_kwargs):
        super().__init__(sample_labeller, **all_kwargs)
        self.source_samples = source_samples  # 保存源图像
        self.blend_images = blend_images  # 保存混合函数


    def augment_sample(self,
                       sample: npt.NDArray[float], # 待增强的目标图像
                       sample_mask: Optional[npt.NDArray[bool]], # 目标图像可选掩码
                       anomaly_corner: npt.NDArray[int], # 放置补丁的左上角坐标
                       anomaly_mask: npt.NDArray[bool], # 补丁的形状掩码
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:

        num_channels = sample.shape[0] # 获取通道数（如1表示灰度图）
        num_dims = len(sample.shape[1:]) # 获取空间维度（如2代表2D图像）

        # Sample source to blend into current sample
        source_sample = random.choice(self.source_samples) # 从源样本库中随机挑选一张图片

        source_sample_shape = np.array(source_sample.shape[1:]) #(256,256) # 获取源图像尺寸


        assert len(source_sample_shape) == num_dims, 'Source and target have different number of spatial dimensions: ' \
                                                     f's-{len(source_sample_shape)}, t-{num_dims}'

        assert source_sample.shape[0] == num_channels, \
            f'Source and target have different number of channels: s-{source_sample.shape[0]}, t-{num_channels}'

        # Compute INVERSE transformation matrix for parameters (rotation, resizing)
        # This is the backwards operation (final source region -> initial source region).
        # 计算逆向变换矩阵（用于定义从最终形状到原始源区域的映射）

        trans_matrix = functools.reduce(lambda m, ds: accumulate_rotation(m,
                                                                          self.rng.uniform(-np.pi / 4, np.pi / 4),
                                                                          ds),
                                        itertools.combinations(range(num_dims), 2),
                                        np.identity(num_dims))
        # 链式应用随机旋转矩阵（-45度到45度之间）

        # Compute effect on corner coords
        target_anomaly_shape = np.array(anomaly_mask.shape)

        corner_coords = np.array(np.meshgrid(*np.stack([np.zeros(num_dims), target_anomaly_shape], axis=-1),
                                             indexing='ij')).reshape(num_dims, 2 ** num_dims)

        trans_corner_coords = trans_matrix @ corner_coords
        min_trans_coords = np.floor(np.min(trans_corner_coords, axis=1))
        max_trans_coords = np.ceil(np.max(trans_corner_coords, axis=1))
        init_grid_shape = max_trans_coords - min_trans_coords

        # Sample scale and clip so that source region isn't too big
        max_scale = np.min(0.8 * source_sample_shape / init_grid_shape)

        # Compute final transformation matrix
        scale_change = 1 + self.rng.exponential(scale=0.1)
        scale_raw = self.rng.choice([scale_change, 1 / scale_change])
        scale = np.minimum(scale_raw, max_scale)

        trans_matrix = accumulate_scaling(trans_matrix, scale)

        # Recompute effect on corner coord
        trans_corner_coords = trans_matrix @ corner_coords
        min_trans_coords = np.floor(np.min(trans_corner_coords, axis=1))
        max_trans_coords = np.ceil(np.max(trans_corner_coords, axis=1))
        final_init_grid_shape = max_trans_coords - min_trans_coords

        # Choose anomaly source location
        final_init_grid_shape = final_init_grid_shape.astype(int)
        min_corner = np.zeros(len(source_sample_shape))
        max_corner = source_sample_shape - final_init_grid_shape

        source_corner = self.rng.integers(min_corner, max_corner, endpoint=True)

        # Extract source
        source_orig = source_sample[get_patch_image_slices(source_corner, tuple(final_init_grid_shape))]


        # Because we computed the backwards transformation we don't need to inverse the matrix
        source_to_blend = np.stack([affine_transform(chan, trans_matrix, offset=-min_trans_coords,
                                                     output_shape=tuple(target_anomaly_shape))
                                    for chan in source_orig])

        spatial_axis = tuple(range(1, len(source_sample.shape)))
        # Spline interpolation can make values fall outside domain, so clip to the original range
        source_to_blend = np.clip(source_to_blend,
                                  source_sample.min(axis=spatial_axis, keepdims=True),
                                  source_sample.max(axis=spatial_axis, keepdims=True))


        # As the blending can alter areas outside the mask, update the mask with any effected areas

        aug_sample = self.blend_images(sample, source_to_blend, anomaly_corner, anomaly_mask)

        sample_slices = get_patch_image_slices(anomaly_corner, tuple(anomaly_mask.shape))
        sample_diff = np.mean(np.abs(sample[sample_slices] - aug_sample[sample_slices]), axis=0)

        anomaly_mask[sample_diff > 0.001] = True
        # Return sample with source blended into it
        return aug_sample



class BaseDeformationTask(BaseTask):

    @abstractmethod
    def compute_mapping(self,
                        sample: npt.NDArray[float],
                        sample_mask: Optional[npt.NDArray[bool]],
                        anomaly_corner: npt.NDArray[int], anomaly_mask: npt.NDArray[bool],
                        anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:
        """
        Returns array of size (*anomaly_mask.shape, len(anomaly_mask.shape)).
        Probably don't need entire sample, but including in for generality.
        :param sample:
        :param sample_mask:
        :param anomaly_corner:
        :param anomaly_mask:
        :param anomaly_intersect_fn:
        :return:
        """

    def augment_sample(self,
                       sample: npt.NDArray[float],
                       sample_mask: Optional[npt.NDArray[bool]],
                       anomaly_corner: npt.NDArray[int],
                       anomaly_mask: npt.NDArray[bool],
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:

        num_channels = sample.shape[0]
        mapping = self.compute_mapping(sample, sample_mask, anomaly_corner, anomaly_mask, anomaly_intersect_fn)
        sample_slices = get_patch_slices(anomaly_corner, tuple(anomaly_mask.shape))

        for chan in range(num_channels):
            sample[chan][sample_slices] = ndimage.map_coordinates(sample[chan][sample_slices],
                                                                  mapping,
                                                                  mode='nearest')
        return sample



class RadialDeformationTask(BaseDeformationTask):

    def __init__(self,
                 sample_labeller: Optional[AnomalyLabeller] = None,
                 deform_factor: Optional[float] = None,
                 deform_centre: Optional[npt.NDArray] = None, **kwargs):

        super().__init__(sample_labeller, **kwargs)
        self.deform_factor = deform_factor
        self.deform_centre = deform_centre
        self.max_anom_prop = 0.6
        self.min_anom_prop = 0.2

    def get_deform_factor(self, def_centre: npt.NDArray[int], anomaly_mask: npt.NDArray[bool]):
        return self.deform_factor if self.deform_factor is not None else 2 ** self.rng.uniform(0.5, 2)

    @abstractmethod
    def compute_new_distance(self, curr_distance: float, max_distance: float, factor: float) -> float:
        """
        Compute new distance for point to be sampled from
        :param curr_distance:
        :param max_distance:
        :param factor:
        """

    def compute_mapping(self,
                        sample: npt.NDArray[float],
                        sample_mask: Optional[npt.NDArray[bool]],
                        anomaly_corner: npt.NDArray[int],
                        anomaly_mask: npt.NDArray[bool],
                        anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:
        # NOTE: This assumes that the shape is convex, will make discontinuities if it's not.

        anomaly_shape = np.array(anomaly_mask.shape)
        num_dims = len(anomaly_shape)

        # Expand so can later be broadcast with (D, N)
        mask_centre = (anomaly_shape - 1) / 2
        exp_mask_centre = np.reshape(mask_centre, (-1, 1))
        # Shape (D, N)
        poss_centre_coords = np.stack(np.nonzero(anomaly_mask))
        def_centre = self.deform_centre if self.deform_centre is not None else \
            poss_centre_coords[:, np.random.randint(poss_centre_coords.shape[1])]

        assert anomaly_mask[tuple(def_centre.round().astype(int))], f'Centre is not within anomaly: {def_centre}'

        # exp_ = expanded
        exp_def_centre = np.reshape(def_centre, (-1, 1))

        # (D, *anomaly_shape)
        mapping = np.stack(np.meshgrid(*[np.arange(s, dtype=float) for s in anomaly_shape], indexing='ij'), axis=0)

        # Ignore pixels on edge of bounding box
        mask_inner_slice = tuple([slice(1, -1)] * num_dims)
        map_inner_slice = tuple([slice(None)] + list(mask_inner_slice))
        # Get all coords and transpose so coord index is last dimension (D, N)
        anomaly_coords = mapping[map_inner_slice][(slice(None), anomaly_mask[mask_inner_slice])]

        all_coords_to_centre = anomaly_coords - exp_def_centre
        all_coords_distance = norm(all_coords_to_centre, axis=0)
        # Ignore zero divided by zero, as we correct it before mapping is returned
        with np.errstate(invalid='ignore'):
            all_coords_norm_dirs = all_coords_to_centre / all_coords_distance

        mask_edge_intersections = anomaly_intersect_fn(exp_def_centre - exp_mask_centre, all_coords_norm_dirs) + exp_mask_centre

        mask_edge_distances = norm(mask_edge_intersections - exp_def_centre, axis=0)

        # Get factor once, so is same for all pixels
        def_factor = self.get_deform_factor(def_centre, anomaly_mask)
        new_coord_distances = self.compute_new_distance(all_coords_distance, mask_edge_distances, def_factor)
        # (D, N)
        new_coords = exp_def_centre + new_coord_distances * all_coords_norm_dirs

        mapping[map_inner_slice][(slice(None), anomaly_mask[mask_inner_slice])] = new_coords

        # Revert centre coordinate, as it will be nan due to the zero magnitude direction vector
        mapping[(slice(None), *def_centre)] = def_centre
        return mapping



class CutPastePatchBlender(BasePatchBlendingTask):

    def __init__(self,
                 source_images: list,
                 Labelber_std: float= 0.2,
                 **kwargs):
        sample_labeller = FlippedGaussianLabeller(Labelber_std)
        source_images=[ np.expand_dims(image,axis=0) if len(image.shape)==2 else image for image in source_images]
        super().__init__(sample_labeller, source_images, cut_paste)
        self.max_anom_prop = 0.6
        self.min_anom_prop = 0.1
        self.anomaly_shape_maker = PerlinPatchMaker()



class SmoothIntensityChangeTask(BaseTask):  # 平滑亮度变化任务类（模拟局部阴影或过曝）

    def __init__(self,
                 intensity_task_scale: float,  # 亮度变化的强度缩放比例
                 sample_labeller: Optional[AnomalyLabeller] = None,
                 **all_kwargs):

        super().__init__(sample_labeller, **all_kwargs)
        self.intensity_task_scale = intensity_task_scale  # 初始化强度
        self.max_anom_prop = 0.8  # 最大异常占比
        self.min_anom_prop = 0.3  # 最小异常占比
        self.anomaly_shape_maker = PerlinPatchMaker()  # 使用Perlin噪声生成补丁形状

    def augment_sample(self,
                       sample: npt.NDArray[float], # 输入的目标图像
                       sample_mask: Optional[npt.NDArray[bool]], # 目标图像掩码
                       anomaly_corner: npt.NDArray[int], # 异常补丁起始位置
                       anomaly_mask: npt.NDArray[bool], # 异常补丁形状掩码
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:

        num_chans = sample.shape[0] # 获取通道数
        sample_shape = sample.shape[1:] # 空间维度尺寸
        num_dims = len(sample_shape) # 维度数

        dist_map = distance_transform_edt(anomaly_mask) # 对异常掩码进行欧率距离变换（计算每个点到边缘的距离）
        min_shape_dim = np.min(sample_shape) # 找到图像的最小边长

        smooth_dist = np.minimum(min_shape_dim * (0.02 + np.random.gamma(3, 0.01)), np.max(dist_map))
        # 随机计算一个平滑过渡的距离参数
        smooth_dist_map = dist_map / smooth_dist # 归一化距离映射，用于平滑混合
        smooth_dist_map[smooth_dist_map > 1] = 1 # 将超出范围的值限制在1以内
        # smooth_dist_map = 1

        anomaly_patch_slices = get_patch_image_slices(anomaly_corner, anomaly_mask.shape)
        # 获取目标图像中对应补丁区域的切片索引

        # anomaly_pixel_stds = np.array([np.std(c[anomaly_mask]) for c in sample[anomaly_patch_slices]])
        # Randomly negate, so some intensity changes are subtractions

        intensity_changes = (self.intensity_task_scale / 2 + np.random.gamma(3, self.intensity_task_scale)) \
            * np.random.choice([1, -1], size=num_chans)
        # 随机计算亮度变化量（可能增加也可能减少）

        intensity_change_map = smooth_dist_map * np.reshape(intensity_changes, [-1] + [1] * num_dims)
        # 将亮度变化量应用于平滑距离图，使得变化在中心最强，边缘平滑衰减

        new_patch = sample[anomaly_patch_slices] + intensity_change_map
        # 在原图的切片区域加上亮度变化

        spatial_axis = tuple(range(1, len(sample.shape)))

        sample[anomaly_patch_slices] = np.clip(new_patch,
                                               sample.min(axis=spatial_axis, keepdims=True),
                                               sample.max(axis=spatial_axis, keepdims=True))
        # 对结果进行裁剪，确保像素值不超出图像原始的最小值和最大值范围

        return sample # 返回增强后的图像



class GaussIntensityChangeTask(BaseTask):  # 高斯亮度变化任务类（模拟由于质地不均产生的纹理型异常）

    def __init__(self,
                 sample_labeller: Optional[AnomalyLabeller] = None,
                 **all_kwargs):

        super().__init__(sample_labeller, **all_kwargs)
        self.max_anom_prop = 0.8  # 最大异常范围
        self.min_anom_prop = 0.3  # 最小异常范围
        self.sigma_bs = [4, 7]  # 高斯模糊的sigma候选值，用于生成不同的纹理精细度
        self.positive_range = [0.4, 0.6]  # 正向亮度变化的期望区间
        self.negative_range = [-0.6, -0.4]  # 负向亮度变化的期望区间
        self.anomaly_shape_maker = PerlinPatchMaker()  # 使用Perlin形状
    def get_predefined_texture(self,
                               mask_shape,  # 生成纹理的尺寸
                               sigma_b,  # 高斯核标准差
                               positive_range=None,
                               negative_range=None,
                               ):  # 获取预定义的基于高斯滤波的随机纹理

        assert (positive_range is not None) or (negative_range is not None)

        random_sample = np.random.randn(mask_shape[0], mask_shape[1]) # 生成随机噪声白图

        random_sample = (random_sample >= 0.0).astype(float)  # 将白图二值化，准备进行高斯平滑

        random_sample = gaussian_filter(random_sample, sigma_b) # 应用高斯模糊，使二值图产生平滑过度的纹理

        random_sample = (random_sample - np.min(random_sample)) / (np.max(random_sample) - np.min(random_sample))
        # 将纹理数据归一化到[0, 1]

        if np.random.uniform(0, 1) <= 0.5: # 50%概率选择变亮
            u_0 = np.random.uniform(positive_range[0], positive_range[1])
        else: # 50%概率选择变暗
            if negative_range is not None:
                u_0 = np.random.uniform(negative_range[0], negative_range[1])
            else:
                u_0 = np.random.uniform(-positive_range[1], -positive_range[0])

        Bj = np.clip(u_0 * random_sample, -1, 1) # 生成带有强度的最终纹理层
        return Bj

    def create_texture(self,sizes): # 调用上述方法创建一个纹理
        texture = self.get_predefined_texture(sizes,
                                         random.choice(self.sigma_bs), # 随机选一个精细度
                                         self.positive_range,
                                         self.negative_range)
        return texture


    def augment_sample(self,
                       sample: npt.NDArray[float], # 目标图
                       sample_mask: Optional[npt.NDArray[bool]],
                       anomaly_corner: npt.NDArray[int],
                       anomaly_mask: npt.NDArray[bool],
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:

        anomaly_mask_copy = anomaly_mask.astype(float) # 将二值掩码转换为浮点型方便计算
        anomaly_patch_slices = get_patch_image_slices(anomaly_corner, anomaly_mask_copy.shape) # 获取切片区域

        texture = self.create_texture(sample.shape[1:]) # 创建全图尺寸的纹理

        while True: # 确保纹理的维度与图像匹配（增加通道维）
            if len(texture.shape)<len(sample.shape):
                texture=np.expand_dims(texture,0)
            else:
                break

        sample = sample / 255.0 # 将图像归一化至[0, 1]
        sigma = np.random.uniform(1, 4) # 随机选择几何模糊程度
        geo_blur = gaussian_filter(anomaly_mask_copy, sigma) # 对掩码应用高斯模糊，模拟有机边缘

        for cha in range(sample.shape[0]): # 遍历通道
            # 将模糊后的掩码区域替换为带有纹理扰动的图像像素
            sample[anomaly_patch_slices][cha] = sample[anomaly_patch_slices][cha] * (1 - anomaly_mask_copy) + \
                                           (sample[anomaly_patch_slices][cha] + texture[anomaly_patch_slices][cha] * geo_blur) * anomaly_mask_copy

        sample = np.clip(sample, a_min=0, a_max=1.0) # 限制像素合法性
        sample = sample * 255.0 # 还原回原始灰度范围
        return sample # 返回增强结果



class IdentityTask(BaseTask):

    def __init__(self,
                 sample_labeller: Optional[AnomalyLabeller] = None,
                 **all_kwargs):

        super().__init__(sample_labeller, **all_kwargs)

    def augment_sample(self,
                       sample: npt.NDArray[float], # aug_sample
                       sample_mask: Optional[npt.NDArray[bool]],# None
                       anomaly_corner: npt.NDArray[int], # anomaly center
                       anomaly_mask: npt.NDArray[bool], # small anomaly mask
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:
        anomaly_mask[:,:] = False
        return sample



class SinkDeformationTask(RadialDeformationTask):
    # y = 1 - (1 - x)^3 (between 0 and 1)
    # -> y = max_d (1 - (1 - curr / max_d) ^ factor)
    # -> y = max_d - (max_d - curr) ^ factor / max_d ^ (factor - 1)

    def compute_new_distance(self, curr_distance: Union[float, npt.NDArray[float]],
                             max_distance: Union[float, npt.NDArray[float]],
                             factor: Union[float, npt.NDArray[float]]) -> Union[float, npt.NDArray[float]]:

        return max_distance - (max_distance - curr_distance) ** factor / max_distance ** (factor - 1)



class SourceDeformationTask(RadialDeformationTask):

    def compute_new_distance(self, curr_distance: Union[float, npt.NDArray[float]],
                             max_distance: Union[float, npt.NDArray[float]],
                             factor: Union[float, npt.NDArray[float]]) -> Union[float, npt.NDArray[float]]:
        # y = x^3 (between 0 and 1)
        # -> y = max_d * (curr / max) ^ factor
        # -> y = curr ^ factor / max_d ^ (factor - 1)   to avoid FP errors
        return curr_distance ** factor / max_distance ** (factor - 1)