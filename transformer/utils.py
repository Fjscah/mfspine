
import numpy as np 
from typing import Optional, Union, Sequence, List, Callable, Tuple
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import map_coordinates
import itertools
import collections
from collections import OrderedDict
import torch 
import os 
from scipy import ndimage as ndi
from batchgenerators.augmentations.utils import resize_segmentation
from skimage.transform import resize



import torch
import random
import numpy as np
import torch.nn.functional as f
import os
import torch.nn as nn
from scipy import ndimage
from sklearn.model_selection import KFold  ## K折交叉验证
import warnings



# def resample_image(image,original_spacing, out_spacing=(1.0, 1.0, 1.0), is_label=False):
#     original_size = image.shape
    
#     out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
#                 int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
#                 int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

#     resample = sitk.ResampleImageFilter()
#     resample.SetOutputSpacing(out_spacing)
#     resample.SetSize(out_size)
#     resample.SetOutputDirection(itk_image.GetDirection())
#     resample.SetOutputOrigin(itk_image.GetOrigin())
#     resample.SetTransform(sitk.Transform())
#     resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

#     if is_label:
#         resample.SetInterpolator(sitk.sitkNearestNeighbor)
#     else:
#         resample.SetInterpolator(sitk.sitkBSpline)

#     return resample.Execute(itk_image)



def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


def resample_tensor_image_label(image, label, size):
 
    if len(image.shape) != 5 or len(label.shape) != 4: 
        print("image or label shape is  err ")
        return
    label = label.float()
    image = nn.functional.interpolate(image, size=size, mode="trilinear", align_corners=False)
    label = torch.unsqueeze(label, dim=1)
    label = nn.functional.interpolate(label, size=size, mode="nearest")
    label = torch.squeeze(label, dim=1).long()
    return image, label

def resample_tensor(tensor_data, size, is_label=False):
    
    if is_label:
        assert len(tensor_data.shape) == 4, "label shape len is 4"
        tensor_data = tensor_data.float()
        tensor_data = torch.unsqueeze(tensor_data, dim=1)
        tensor_data = nn.functional.interpolate(tensor_data, size=size, mode="nearest")
        tensor_data = torch.squeeze(tensor_data, dim=1).long()
    else:
        assert len(tensor_data.shape) == 5, 'image shape len is 5'
        tensor_data = nn.functional.interpolate(tensor_data, size=size, mode="trilinear", align_corners=False)

    
    return tensor_data


def get_kfold_data(data_paths, n_splits, total_num, shuffle=False):
    X = np.arange(total_num)
    kfold = KFold(n_splits=n_splits, shuffle=shuffle)  ## kfold为KFolf类的一个对象
    return_res = []
    for a, b in kfold.split(X):
        fold_train = []
        fold_val = []
        for i in a:
            fold_train.append(data_paths[i])
        for j in b:
            fold_val.append(data_paths[j])
        return_res.append({"train_data": fold_train, "val_data": fold_val})

    return return_res

def write_res_line(w_path, content):
    with open(w_path, "a+", encoding="utf-8") as f:
        if type(content) is list:
            for c in content:
                f.write("写入结果为：")
                f.write(str(c))
                f.write("\n")
        else :
            f.write("写入结果为：")
            f.write(str(content))
            f.write("\n")

def softmin(x, dim=-1, t=1):
    m = nn.Softmin(dim)

    return  m(x / t)

def compute_orthogonal_loss(net_1, net_2):
    device = next(net_1.parameters()).device
    orthogonal_loss = 0.0
    n = 0
    for name, _ in net_1.named_parameters():
        weight_1 = net_1.state_dict()[name]
        weight_2 = net_2.state_dict()[name]

        if len(weight_1.shape) != 5:
            # print("非卷积核权重，跳过")
            continue
        n += 1
        kernel_size = weight_1.shape[0]
        weight_1 = weight_1.view((kernel_size, -1))
        weight_2 = weight_2.view((kernel_size, -1))

        weight_1 = f.normalize(weight_1, dim=-1, p=2)
        weight_2 = f.normalize(weight_2, dim=-1, p=2)
        sim_matrix = torch.matmul(weight_1, weight_2.t())

        err = 0.5 * (sim_matrix ** 2).sum()
        # mask = torch.eye(*sim_matrix.shape, device=device)
        # err = (sim_matrix**2 * mask).sum()
        # print(f"err is {err}")
        # print(f"sim matrix is {sim_matrix * mask}")

        orthogonal_loss += err
    # orthogonal_loss = orthogonal_loss / n # 求平均
    return orthogonal_loss


def set_seed(seed: int, use_deterministic_algorithms=None):
   

    seed = int(seed)
    torch.manual_seed(seed)

    global _seed
    _seed = seed
    random.seed(seed)
    np.random.seed(seed)

    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
   
    if use_deterministic_algorithms is not None:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(use_deterministic_algorithms)
        elif hasattr(torch, "set_deterministic"):
            torch.set_deterministic(use_deterministic_algorithms)  # type: ignore
        else:
            warnings.warn("use_deterministic_algorithms=True, but PyTorch version is too old to set the mode.")


def compute_dice(input, target):
    eps = 0.0001
    # input 是经过了sigmoid 之后的输出。
    input = (input > 0.5).float()
    target = (target > 0.5).float()
    if target.sum() == 0:
        return 1.0

    inter = torch.sum(target.view(-1) * input.view(-1)) + eps

    # print(self.inter)
    union = torch.sum(input) + torch.sum(target) + eps

    t = (2 * inter.float()) / union.float()
    if int(t.item()) == 2:
        return 1.0
    return t.item()


def compute_dice_muti_class(input, target):

    ## 此时是多个类别应该得分开算dice
    class_num = input.shape[1]
    input = input.argmax(dim=1)
    # print(input.shape)
    dice = []
    for i in range(class_num):
        if i == 0:
            continue
        label = torch.zeros_like(target)
        input_single = torch.zeros_like(target)
        label[target == i] = 1
        input_single[input == i] = 1
        dice.append(compute_dice(input_single, label))
    return dice


def compute_iou(pred, label):
    # 计算iou loss
    inter = (pred * label).sum()
    union = torch.logical_or(pred, label)
    union = union.sum()

    return ((inter + 1e-6) /(union + 1e-6)).item()


def compute_3d_metric(pred_3d, label):
    ## 计算多个指标 3d
    # pred_3d_sig = (pred_3d_sig > 0.5).float()
    seg_inv, gt_inv = torch.logical_not(pred_3d), torch.logical_not(label)
    true_pos = float(torch.logical_and(pred_3d, label).sum())  # float for division
    true_neg = torch.logical_and(seg_inv, gt_inv).sum()
    false_pos = torch.logical_and(pred_3d, gt_inv).sum()
    false_neg = torch.logical_and(seg_inv, label).sum()

    # 然后根据公式分别计算出这几种指标
    prec = true_pos / (true_pos + false_pos + 1e-6)
    rec = true_pos / (true_pos + false_neg + 1e-6)
    specificity = true_neg / (true_neg + false_neg + 1e-6)

    return prec.item(), rec.item(), specificity.item()


def segmenation_metric(pred, target):
    class_num = pred.shape[1]
    pred = pred.argmax(dim=1)
    # print(input.shape)
    metric = []

    for i in range(class_num):
        if i == 0:
            continue
        label = torch.zeros_like(target)
        input_single = torch.zeros_like(target)
        label[target == i] = 1
        input_single[pred == i] = 1
        dice = compute_dice(input_single, label)
        prec, rec, specificity = compute_3d_metric(input_single, label)
        iou = compute_iou(input_single, label)

        metric.append([dice, prec, rec, specificity, iou])
    # print(metric)
    return metric


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def resample_image_array_size(image_array, out_size, order=3):
    #Bilinear interpolation would be order=1,
    # nearest is order=0,
    # and cubic is the default (order=3).
    real_resize = np.array(out_size) / image_array.shape

    new_volume = ndimage.zoom(image_array, zoom=real_resize, order=order)
    return new_volume





def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, order_z=0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param cval:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order,
                                                       **kwargs))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                                   mode='nearest')[None])
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None])
                else:
                    reshaped_final_data.append(reshaped_data[None])
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None])
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data

def issequenceiterable(obj) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    if isinstance(obj, torch.Tensor):
        return int(obj.dim()) > 0  # a 0-d tensor is not iterable
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, (str, bytes))

def ensure_tuple_rep(tup, dim: int):
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

    Examples::

        >>> ensure_tuple_rep(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_rep(None, 3)
        (None, None, None)
        >>> ensure_tuple_rep('test', 3)
        ('test', 'test', 'test')
        >>> ensure_tuple_rep([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_rep(range(3), 3)
        (0, 1, 2)
        >>> ensure_tuple_rep([1, 2], 3)
        ValueError: Sequence must have length 3, got length 2.

    """
    if isinstance(tup, torch.Tensor):
        tup = tup.detach().cpu().numpy()
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")

# def compute_divisible_spatial_size(spatial_shape: Sequence[int], k: Union[Sequence[int], int]):
#     """
#     Compute the target spatial size which should be divisible by `k`.

#     Args:
#         spatial_shape: original spatial shape.
#         k: the target k for each spatial dimension.
#             if `k` is negative or 0, the original size is preserved.
#             if `k` is an int, the same `k` be applied to all the input spatial dimensions.

#     """
#     k = fall_back_tuple(k, (1,) * len(spatial_shape))
#     new_size = []
#     for k_d, dim in zip(k, spatial_shape):
#         new_dim = int(np.ceil(dim / k_d) * k_d) if k_d > 0 else dim
#         new_size.append(new_dim)

#     return new_size

class Pad:
    """
    Perform padding for a given an amount of padding in each dimension.
    If input is `torch.Tensor`, `torch.nn.functional.pad` will be used, otherwise, `np.pad` will be used.

    Args:
        to_pad: the amount to be padded in each dimension [(low_H, high_H), (low_W, high_W), ...].
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """
    def __init__(
        self,
        to_pad: List[Tuple[int, int]],
        mode = "constant",
        **kwargs,
    ) -> None:
        self.to_pad = to_pad
        self.mode = mode
        self.kwargs = kwargs

    @staticmethod
    def _np_pad(img: np.ndarray, all_pad_width, mode, **kwargs) -> np.ndarray:
        return np.pad(img, all_pad_width, mode=mode, **kwargs)  # type: ignore

  
    def __call__(
        self, img, mode = None
    ):
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first and
                padding doesn't apply to the channel dim.
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"`` or ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to `self.mode`.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

        """
        if not np.asarray(self.to_pad).any():
            # all zeros, skip padding
            return img
        mode = convert_pad_mode(dst=img, mode=mode or self.mode).value
        pad = self._np_pad
        return pad(img, self.to_pad, mode, **self.kwargs)  # type: ignore


def convert_pad_mode(dst, mode):
    """
    Utility to convert padding mode between numpy array and PyTorch Tensor.

    Args:
        dst: target data to convert padding mode for, should be numpy array or PyTorch Tensor.
        mode: current padding mode.

    """
    if isinstance(dst, np.ndarray):
        if mode == "circular":
            mode = "wrap"
        if mode == "replicate":
            mode = "edge"
        return mode

    raise ValueError(f"unsupported data type: {type(dst)}.")

def is_positive(img):
    """
    Returns a boolean version of `img` where the positive values are converted into True, the other values are False.
    """
    return img > 0


def where(condition, x=None, y=None):
    """
    Note that `torch.where` may convert y.dtype to x.dtype.
    """
    if isinstance(condition, np.ndarray):
        if x is not None:
            result = np.where(condition, x, y)
        else:
            result = np.where(condition)
    else:
        if x is not None:
            x = torch.as_tensor(x, device=condition.device)
            y = torch.as_tensor(y, device=condition.device, dtype=x.dtype)
            result = torch.where(condition, x, y)
        else:
            result = torch.where(condition)  # type: ignore
    return result

def generate_spatial_bounding_box(
    img,
    select_fn: Callable = is_positive,
    channel_indices = None,
    margin: Union[Sequence[int], int] = 0,
) -> Tuple[List[int], List[int]]:
    """
    generate the spatial bounding box of foreground in the image with start-end positions.
    Users can define arbitrary function to select expected foreground from the whole image or specified channels.
    And it can also add margin to every dim of the bounding box.
    The output format of the coordinates is:

        [1st_spatial_dim_start, 2nd_spatial_dim_start, ..., Nth_spatial_dim_start],
        [1st_spatial_dim_end, 2nd_spatial_dim_end, ..., Nth_spatial_dim_end]

    The bounding boxes edges are aligned with the input image edges.
    This function returns [-1, -1, ...], [-1, -1, ...] if there's no positive intensity.

    Args:
        img: source image to generate bounding box from.
        select_fn: function to select expected foreground, default is to select values > 0.
        channel_indices: if defined, select foreground only on the specified channels
            of image. if None, select foreground on the whole image.
        margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
    """
    data = img[list(channel_indices)] if channel_indices is not None else img
    data = select_fn(data).any(0)
    ndim = len(data.shape)
    margin = ensure_tuple_rep(margin, ndim)
    for m in margin:
        if m < 0:
            raise ValueError("margin value should not be negative number.")

    box_start = [0] * ndim
    box_end = [0] * ndim

    for di, ax in enumerate(itertools.combinations(reversed(range(ndim)), ndim - 1)):
        dt = data
        if len(ax) != 0:
            dt = np.any(dt, ax)

        if not dt.any():
            # if no foreground, return all zero bounding box coords
            return [0] * ndim, [0] * ndim

        arg_max = where(dt == dt.max())[0]
        min_d = max(arg_max[0] - margin[di], 0)
        max_d = arg_max[-1] + margin[di] + 1

        box_start[di] = min_d.detach().cpu().item() if isinstance(min_d, torch.Tensor) else min_d  # type: ignore
        box_end[di] = max_d.detach().cpu().item() if isinstance(max_d, torch.Tensor) else max_d  # type: ignore

    return box_start, box_end

def create_zero_centered_coordinate_mesh(shape):
        tmp = tuple([np.arange(i) for i in shape])
        coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
        for d in range(len(shape)):
            coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
        
        return coords

def elastic_deform_coordinates(coordinates, alpha, sigma, random_state):
    n_dim = len(coordinates)
    offsets = []
    for _ in range(n_dim):
        offsets.append(
            gaussian_filter((random_state.random(coordinates.shape[1:]) * 2 - 1), sigma, mode="constant", cval=0) * alpha)
    offsets = np.array(offsets)
    indices = offsets + coordinates
    return indices

def interpolate_img(img, coords, order=3, mode='nearest', cval=0.0, is_seg=False):
    if is_seg and order != 0:
        unique_labels = np.unique(img)
        result = np.zeros(coords.shape[1:], img.dtype)
        for i, c in enumerate(unique_labels):
            res_new = map_coordinates((img == c).astype(float), coords, order=order, mode=mode, cval=cval)
            result[res_new >= 0.5] = c
        return result
    else:
        return map_coordinates(img.astype(float), coords, order=order, mode=mode, cval=cval).astype(img.dtype)

def scale_coords(coords, scale):
    if isinstance(scale, (tuple, list, np.ndarray)):
        assert len(scale) == len(coords)
        for i in range(len(scale)):
            coords[i] *= scale[i]
    else:
        coords *= scale
    return coords

def correct_crop_centers(
    centers: List[np.ndarray], spatial_size: Union[Sequence[int], int], label_spatial_shape: Sequence[int]
) -> List[np.ndarray]:
    """
    Utility to correct the crop center if the crop size is bigger than the image size.
    Args:
        ceters: pre-computed crop centers, will correct based on the valid region.
        spatial_size: spatial size of the ROIs to be sampled.
        label_spatial_shape: spatial shape of the original label data to compare with ROI.
    """
    spatial_size = spatial_size
    default=label_spatial_shape
    if not (np.subtract(label_spatial_shape, spatial_size) >= 0).all():
        raise ValueError("The size of the proposed random crop ROI is larger than the image size.")

    # Select subregion to assure valid roi
    valid_start = np.floor_divide(spatial_size, 2)
    # add 1 for random
    valid_end = np.subtract(label_spatial_shape + np.array(1), spatial_size / np.array(2)).astype(np.uint16)
    # int generation to have full range on upper side, but subtract unfloored size/2 to prevent rounded range
    # from being too high
    for i, valid_s in enumerate(valid_start):
        # need this because np.random.randint does not work with same start and end
        if valid_s == valid_end[i]:
            valid_end[i] += 1

    for i, c in enumerate(centers):
        center_i = c
        if c < valid_start[i]:
            center_i = valid_start[i]
        if c >= valid_end[i]:
            center_i = valid_end[i] - 1
        centers[i] = center_i

    return centers

def generate_pos_neg_label_crop_centers(
    spatial_size: Union[Sequence[int], int],
    num_samples: int,
    pos_ratio: float,
    label_spatial_shape: Sequence[int],
    fg_indices: np.ndarray,
    bg_indices: np.ndarray,
    rand_state: Optional[np.random.RandomState] = None,
) -> List[List[np.ndarray]]:
    """
    Generate valid sample locations based on the label with option for specifying foreground ratio
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]
    Args:
        spatial_size: spatial size of the ROIs to be sampled.
        num_samples: total sample centers to be generated.
        pos_ratio: ratio of total locations generated that have center being foreground.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        fg_indices: pre-computed foreground indices in 1 dimension.
        bg_indices: pre-computed background indices in 1 dimension.
        rand_state: numpy randomState object to align with other modules.
    Raises:
        ValueError: When the proposed roi is larger than the image.
        ValueError: When the foreground and background indices lengths are 0.
    """
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore

    centers = []
    fg_indices, bg_indices = np.asarray(fg_indices), np.asarray(bg_indices)
    if fg_indices.size == 0 and bg_indices.size == 0:
        raise ValueError("No sampling location available.")

    if fg_indices.size == 0 or bg_indices.size == 0:
        print(
            f"N foreground {len(fg_indices)}, N  background {len(bg_indices)},"
            "unable to generate class balanced samples."
        )
        pos_ratio = 0 if fg_indices.size == 0 else 1

    for _ in range(num_samples):
        indices_to_use = fg_indices if rand_state.rand() < pos_ratio else bg_indices
        random_int = rand_state.randint(len(indices_to_use))
        center = np.unravel_index(indices_to_use[random_int], label_spatial_shape)
        # shift center to range of valid centers
        center_ori = list(center)
        centers.append(correct_crop_centers(center_ori, spatial_size, label_spatial_shape))

    return centers



def augment_gamma(data_sample, gamma_range=(0.5, 2), epsilon=1e-7, per_channel=False,
                  retain_stats: Union[bool, Callable[[], bool]] = False):
    
    if not per_channel:
        retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
        if retain_stats_here:
            mn = data_sample.mean()
            sd = data_sample.std()
        if gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats_here:
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn
    else:
        for c in range(data_sample.shape[0]):
            retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
            if retain_stats_here:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats_here:
                data_sample[c] = data_sample[c] - data_sample[c].mean()
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
                data_sample[c] = data_sample[c] + mn
    
    return data_sample


def augment_mirroring(sample_data, random_state, sample_seg=None, axes=(0, 1, 2)):
    if len(sample_seg.shape) == 3:
        # add a dimension
        sample_seg = np.expand_dims(sample_seg, axis=0)

    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]")
    if 0 in axes and random_state.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and random_state.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if random_state.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    if len(sample_seg.shape) == 4:
        sample_seg = np.squeeze(sample_seg, axis=0)
        
    return sample_data, sample_seg