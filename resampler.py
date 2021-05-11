import torch
import numpy as np
from numpy.linalg import matrix_rank

def safe_gather_nd(params, indices):
    """Gather slices from params into a Tensor with shape specified by indices.
    Similar functionality to tf.gather_nd with difference: when index is out of
    bound, always return 0.
    Args:
    params: A Tensor. The tensor from which to gather values.
    indices: A Tensor. Must be one of the following types: int32, int64. Index
        tensor.
    Returns:
    A Tensor. Has the same type as params. Values from params gathered from
    specified indices (if they exist) otherwise zeros, with shape
    indices.shape[:-1] + params.shape[indices.shape[-1]:].
    """
    params_shape = params.shape
    indices_shape = indices.shape
    slice_dimensions = indices_shape[-1]

    max_index = params_shape[:slice_dimensions] - 1
    min_index = torch.zeros_like(max_index, dtype=torch.int32)

    clipped_indices = torch.clamp(indices, min_index, max_index)

    # Check whether each component of each index is in range [min, max], and
    # allow an index only if all components are in range:
    mask = torch.all(
        torch.logical_and(indices >= min_index, indices <= max_index), dim=1)
    mask = torch.unsqueeze(mask, -1)

    return (mask.type(dtype=params.dtype) *
            torch_gather_nd(params, clipped_indices))

def torch_gather_nd(params, indices):
    '''
    the input indices must be a 2d tensor in the form of [[a,b,..,c],...], 
    which represents the location of the elements.
    '''
    indices = indices.type(torch.LongTensor)
    params = params.view(params.shape[0], 128, 416, -1)
    output = torch.zeros_like(params, device=params.device)
    # new_output = torch.zeros_like(params, device=params.device)
    ind_i = torch.arange(0, end=indices.shape[0], dtype=torch.long)
    ind_j = torch.arange(0, end=indices.shape[1], dtype=torch.long)
    ind_k = torch.arange(0, end=indices.shape[2], dtype=torch.long)
    ind = torch.meshgrid(ind_i, ind_j, ind_k)
    index = torch.unbind(indices[ind], dim=-1)
    output[index] = params[index]
    # for i in range(indices.shape[0]):
    #     for j in range(indices.shape[1]):
    #         for k in range(indices.shape[2]):
    #                 new_output[tuple(indices[i,j,k])] = params[tuple(indices[i,j,k])]
    # print(torch.all(torch.eq(new_output, output)))
    return output

def resampler_with_unstacked_warp(data,
                                  warp_x,
                                  warp_y,
                                  safe=True,
                                  name='resampler'):
    """Resamples input data at user defined coordinates.
    Args:
    data: Tensor of shape `[batch_size, data_height, data_width,
        data_num_channels]` containing 2D data that will be resampled.
    warp_x: Tensor of shape `[batch_size, dim_0, ... , dim_n]` containing the x
        coordinates at which resampling will be performed.
    warp_y: Tensor of the same shape as warp_x containing the y coordinates at
        which resampling will be performed.
    safe: A boolean, if True, warp_x and warp_y will be clamped to their bounds.
        Disable only if you know they are within bounds, otherwise a runtime
        exception will be thrown.
    name: Optional name of the op.
    Returns:
        Tensor of resampled values from `data`. The output tensor shape is
    `[batch_size, dim_0, ... , dim_n, data_num_channels]`.
    Raises:
    ValueError: If warp_x, warp_y and data have incompatible shapes.
    """
    warp_x = warp_x
    warp_y = warp_y
    data = data
    if not warp_x.shape == warp_y.shape:
        raise ValueError(
            'warp_x and warp_y are of incompatible shapes: %s vs %s ' %
            (str(warp_x.shape), str(warp_y.shape)))
    warp_shape = torch.tensor(list(warp_x.shape))
    if warp_x.shape[0] != data.shape[0]:
        raise ValueError(
            '\'warp_x\' and \'data\' must have compatible first '
            'dimension (batch size), but their shapes are %s and %s ' %
            (str(warp_x.shape[0]), str(data.shape[0])))
    # Compute the four points closest to warp with integer value.
    warp_floor_x = torch.floor(warp_x)
    warp_floor_y = torch.floor(warp_y)
    # Compute the weight for each point.
    right_warp_weight = warp_x - warp_floor_x
    down_warp_weight = warp_y - warp_floor_y

    warp_floor_x = warp_floor_x.int()
    warp_floor_y = warp_floor_y.int()
    warp_ceil_x = torch.ceil(warp_x).int()
    warp_ceil_y = torch.ceil(warp_y).int()

    left_warp_weight = 1 - right_warp_weight
    up_warp_weight = 1 - down_warp_weight

    # Extend warps from [batch_size, dim_0, ... , dim_n, 2] to
    # [batch_size, dim_0, ... , dim_n, 3] with the first element in last
    # dimension being the batch index.

    # A shape like warp_shape but with all sizes except the first set to 1:
    warp_batch_shape = torch.cat(
        (warp_shape[0:1], torch.ones_like(warp_shape[1:])), dim=0)

    warp_batch = torch.arange(start=0, end=warp_shape[0], 
                dtype=torch.int32, device=warp_y.device).reshape(tuple(warp_batch_shape))
    # Broadcast to match shape:
    warp_batch = torch.add(warp_batch, 
                    torch.zeros_like(warp_y, dtype=torch.int32, device=warp_y.device))
    left_warp_weight = torch.unsqueeze(left_warp_weight, -1)
    down_warp_weight = torch.unsqueeze(down_warp_weight, -1)
    up_warp_weight = torch.unsqueeze(up_warp_weight, -1)
    right_warp_weight = torch.unsqueeze(right_warp_weight, -1)

    up_left_warp = torch.stack([warp_batch, warp_floor_y, warp_floor_x], dim=-1)
    up_right_warp = torch.stack([warp_batch, warp_floor_y, warp_ceil_x], dim=-1)
    down_left_warp = torch.stack([warp_batch, warp_ceil_y, warp_floor_x], dim=-1)
    down_right_warp = torch.stack([warp_batch, warp_ceil_y, warp_ceil_x], dim=-1)
    
    def gather_nd(params, indices):
        return (safe_gather_nd if safe else torch_gather_nd)(params, indices)

    # gather data then take weighted average to get resample result.
    result = (
        (gather_nd(data, up_left_warp) * left_warp_weight +
            gather_nd(data, up_right_warp) * right_warp_weight) * up_warp_weight +
        (gather_nd(data, down_left_warp) * left_warp_weight +
            gather_nd(data, down_right_warp) * right_warp_weight) *
        down_warp_weight)
    result = result.view(warp_x.shape[0], -1, warp_x.shape[1], warp_x.shape[2])
    return result