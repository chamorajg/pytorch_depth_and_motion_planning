from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import resampler
import transform_utils

def multiply_no_nan(a, b):
    res = torch.mul(a, b)
    res[res != res] = 0
    return res 

def rgbd_consistency_loss(frame1transformed_depth,
                          frame1rgb,
                          frame2depth,
                          frame2rgb,
                          validity_mask=None):
    """Computes a loss that penalizes RGBD inconsistencies between frames.
    This function computes 3 losses that penalize inconsistencies between two
    frames: depth, RGB, and structural similarity. It IS NOT SYMMETRIC with
    respect to both frames. In particular, to address occlusions, it only
    penalizes depth and RGB inconsistencies at pixels where frame1 is closer to
    the camera than frame2 (Why? see https://arxiv.org/abs/1904.04998). Therefore
    the intended usage pattern is running it twice - second time with the two
    frames swapped.
    Args:
    frame1transformed_depth: A transform_depth_map.TransformedDepthMap object
        representing the depth map of frame 1 after it was motion-transformed to
        frame 2, a motion transform that accounts for all camera and object motion
        that occurred between frame1 and frame2. The tensors inside
        frame1transformed_depth are of shape [B, H, W].
    frame1rgb: A torch.Tensor of shape [B, C, H, W] containing the RGB image at
        frame1.
    frame2depth: A torch.Tensor of shape [B, H, W] containing the depth map at
        frame2.
    frame2rgb: A torch.Tensor of shape [B, C, H, W] containing the RGB image at
        frame2.
    validity_mask: a torch.Tensor of a floating point type and a shape of
        [B, H, W, 1] containing a validity mask.
    Returns:
    A dicionary from string to torch.Tensor, with the following entries:
        depth_error: A tf scalar, the depth mismatch error between the two frames.
        rgb_error: A tf scalar, the rgb mismatch error between the two frames.
        ssim_error: A tf scalar, the strictural similarity mismatch error between
        the two frames.
        depth_proximity_weight: A torch.Tensor of shape [B, H, W], representing a
        function that peaks (at 1.0) for pixels where there is depth consistency
        between the two frames, and is small otherwise.
        frame1_closer_to_camera: A torch.Tensor of shape [B, H, W, 1], a mask that is
        1.0 when the depth map of frame 1 has smaller depth than frame 2.
    """
    frame2rgbd = torch.cat(
        (frame2rgb, frame2depth), dim=1)
    frame2rgbd_resampled = resampler.resampler_with_unstacked_warp(
        frame2rgbd,
        frame1transformed_depth.pixel_x,
        frame1transformed_depth.pixel_y,
        safe=False)
    frame2rgb_resampled, frame2depth_resampled = torch.split(
        frame2rgbd_resampled, [3, 1], dim=1)
    frame2depth_resampled = torch.squeeze(frame2depth_resampled, dim=1)

    # f1td.depth is the predicted depth at [pixel_y, pixel_x] for frame2. Now we
    # generate (by interpolation) the actual depth values for frame2's depth, at
    # the same locations, so that we can compare the two depths.

    # We penalize inconsistencies between the two frames' depth maps only if the
    # transformed depth map (of frame 1) falls closer to the camera than the
    # actual depth map (of frame 2). This is intended for avoiding penalizing
    # points that become occluded because of the transform.
    # So what about depth inconsistencies where frame1's depth map is FARTHER from
    # the camera than frame2's? These will be handled when we swap the roles of
    # frame 1 and 2 (more in https://arxiv.org/abs/1904.04998).
    frame1_closer_to_camera = torch.logical_and(
            frame1transformed_depth.mask,
            torch.lt(frame1transformed_depth.depth, frame2depth_resampled)).type(
                torch.FloatTensor).to(device=frame2depth_resampled.device)
    frames_l1_diff = torch.abs(frame2depth_resampled - frame1transformed_depth.depth)
    if validity_mask is not None:
        frames_l1_diff = frames_l1_diff * torch.squeeze(validity_mask, dim=1)

    depth_error = torch.mean(
        multiply_no_nan(frames_l1_diff, frame1_closer_to_camera)) # TODO

    frames_rgb_l1_diff = torch.abs(frame2rgb_resampled - frame1rgb)
    if validity_mask is not None:
        frames_rgb_l1_diff = frames_rgb_l1_diff * validity_mask
    rgb_error = multiply_no_nan(
        frames_rgb_l1_diff, torch.unsqueeze(frame1_closer_to_camera, 1)) # TODO
    rgb_error = torch.mean(rgb_error)

    # We generate a weight function that peaks (at 1.0) for pixels where when the
    # depth difference is less than its standard deviation across the frame, and
    # fall off to zero otherwise. This function is used later for weighing the
    # structural similarity loss term. We only want to demand structural
    # similarity for surfaces that are close to one another in the two frames.
    depth_error_second_moment = _weighted_average(
        torch.square(frame2depth_resampled - frame1transformed_depth.depth),
        frame1_closer_to_camera) + 1e-4
    depth_proximity_weight = multiply_no_nan(
        depth_error_second_moment /
        (torch.square(frame2depth_resampled - frame1transformed_depth.depth) +
        depth_error_second_moment), frame1transformed_depth.mask.type(
            torch.FloatTensor).to(device=frame1transformed_depth.depth.device)) # TODO

    if validity_mask is not None:
        depth_proximity_weight = depth_proximity_weight * torch.squeeze(
            validity_mask, dim=1)

    # If we don't stop the gradient training won't start. The reason is presumably
    # that then the network can push the depths apart instead of seeking RGB
    # consistency.
    depth_proximity_weight = depth_proximity_weight.detach()

    ssim_error, avg_weight = weighted_ssim(
        frame2rgb_resampled,
        frame1rgb,
        depth_proximity_weight,
        c1=float('inf'),  # These values of c1 and c2 seemed to work better than
        c2=9e-6)  # defaults. TODO(gariel): Make them parameters rather
    # than hard coded.
    ssim_error_mean = torch.mean(
        multiply_no_nan(ssim_error, avg_weight)) # TODO

    endpoints = {
        'depth_error': depth_error,
        'rgb_error': rgb_error,
        'ssim_error': ssim_error_mean,
        'depth_proximity_weight': depth_proximity_weight,
        'frame1_closer_to_camera': frame1_closer_to_camera
    }
    return endpoints


def motion_field_consistency_loss(frame1transformed_pixelx,
                                  frame1transformed_pixely, mask, rotation1,
                                  translation1, rotation2, translation2):
    """Computes a cycle consistency loss between two motion maps.
    Given two rotation and translation maps (of two frames), and a mapping from
    one frame to the other, this function assists in imposing that the fields at
    frame 1 represent the opposite motion of the ones in frame 2.
    In other words: At any given pixel on frame 1, if we apply the translation and
    rotation designated at that pixel, we land on some pixel in frame 2, and if we
    apply the translation and rotation designated there, we land back at the
    original pixel at frame 1.
    Args:
    frame1transformed_pixelx: A torch.Tensor of shape [B, H, W] representing the
        motion-transformed x-location of each pixel in frame 1.
    frame1transformed_pixely: A torch.Tensor of shape [B, H, W] representing the
        motion-transformed y-location of each pixel in frame 1.
    mask: A torch.Tensor of shape [B, H, W, 2] expressing the weight of each pixel
        in the calculation of the consistency loss.
    rotation1: A torch.Tensor of shape [B, 3] representing rotation angles.
    translation1: A torch.Tensor of shape [B, H, W, 3] representing translation
        vectors.
    rotation2: A torch.Tensor of shape [B, 3] representing rotation angles.
    translation2: A torch.Tensor of shape [B, H, W, 3] representing translation
        vectors.
    Returns:
    A dicionary from string to torch.Tensor, with the following entries:
        rotation_error: A tf scalar, the rotation consistency error.
        translation_error: A tf scalar, the translation consistency error.
    """

    translation2resampled = resampler.resampler_with_unstacked_warp(
        translation2,
        frame1transformed_pixelx.detach(),
        frame1transformed_pixely.detach(),
        safe=False)
    translation2resampled = translation2resampled.view(-1, 128, 416, 3)
    rotation1field = _expand_dims_twice(rotation1, -2).expand(
                    translation1.shape)
    rotation2field = _expand_dims_twice(rotation2, -2).expand(
                    translation2.shape)
    rotation1matrix = transform_utils.matrix_from_angles(rotation1field)
    rotation2matrix = transform_utils.matrix_from_angles(rotation2field)

    rot_unit, trans_zero = transform_utils.combine(rotation2matrix,
                                                    translation2resampled,
                                                    rotation1matrix, translation1)
    eye = torch.eye(3).to(device=rot_unit.device) #batch_shape=rot_unit.shape[:-2]
    for i in range(len(rot_unit.shape[:-2])):
        eye = eye.unsqueeze(0)
    eye = eye.repeat(*rot_unit.shape[:-2], 1, 1)
    

    # We normalize the product of rotations by the product of their norms, to make
    # the loss agnostic of their magnitudes, only wanting them to be opposite in
    # directions. Otherwise the loss has a tendency to drive the rotations to
    # zero.
    rot_error = torch.mean(torch.square(rot_unit - eye), dim=(3, 4))
    rot1_scale = torch.mean(torch.square(rotation1matrix - eye), dim=(3, 4))
    rot2_scale = torch.mean(torch.square(rotation2matrix - eye), dim=(3, 4))
    rot_error = rot_error / (1e-24 + rot1_scale + rot2_scale)
    rotation_error = torch.mean(rot_error)

    def norm(x):
        return torch.sum(torch.square(x), dim=-1)

    # Here again, we normalize by the magnitudes, for the same reason.
    translation_error = torch.mean(torch.mul(
        mask, norm(trans_zero) /
        (1e-24 + norm(translation1) + norm(translation2resampled))))

    return {
        'rotation_error': rotation_error,
        'translation_error': translation_error
    }


def rgbd_and_motion_consistency_loss(frame1transformed_depth,
                                     frame1rgb,
                                     frame2depth,
                                     frame2rgb,
                                     rotation1,
                                     translation1,
                                     rotation2,
                                     translation2,
                                     validity_mask=None):
    """A helper that bundles rgbd and motion consistency losses together."""
    endpoints = rgbd_consistency_loss(
        frame1transformed_depth,
        frame1rgb,
        frame2depth,
        frame2rgb,
        validity_mask=validity_mask)
    # We calculate the loss only for when frame1transformed_depth is closer to the
    # camera than frame2 (occlusion-awareness). See explanation in
    # rgbd_consistency_loss above.
    mask = endpoints['frame1_closer_to_camera']
    if validity_mask is not None:
        mask = mask * torch.squeeze(validity_mask, dim=1)
    endpoints.update(
        motion_field_consistency_loss(frame1transformed_depth.pixel_x,
                                    frame1transformed_depth.pixel_y, mask,
                                    rotation1, translation1, rotation2,
                                    translation2))
    return endpoints


def weighted_ssim(x, y, weight, c1=0.01**2, c2=0.03**2, weight_epsilon=0.01):
    """Computes a weighted structured image similarity measure.
    See https://en.wikipedia.org/wiki/Structural_similarity#Algorithm. The only
    difference here is that not all pixels are weighted equally when calculating
    the moments - they are weighted by a weight function.
    Args:
    x: A torch.Tensor representing a batch of images, of shape [B, C, H, W].
    y: A torch.Tensor representing a batch of images, of shape [B, C, H, W].
    weight: A torch.Tensor of shape [B, H, W], representing the weight of each
        pixel in both images when we come to calculate moments (means and
        correlations).
    c1: A floating point number, regularizes division by zero of the means.
    c2: A floating point number, regularizes division by zero of the second
        moments.
    weight_epsilon: A floating point number, used to regularize division by the
        weight.
    Returns:
    A tuple of two torch.Tensors. First, of shape [B, H-2, W-2, C], is scalar
    similarity loss oer pixel per channel, and the second, of shape
    [B, H-2. W-2, 1], is the average pooled `weight`. It is needed so that we
    know how much to weigh each pixel in the first tensor. For example, if
    `'weight` was very small in some area of the images, the first tensor will
    still assign a loss to these pixels, but we shouldn't take the result too
    seriously.
    """
    if c1 == float('inf') and c2 == float('inf'):
        raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                            'likely unintended.')
    weight = torch.unsqueeze(weight, 1)
    average_pooled_weight = _avg_pool3x3(weight)
    weight_plus_epsilon = weight + weight_epsilon
    inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

    def weighted_avg_pool3x3(z):
        wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
        return wighted_avg * inverse_average_pooled_weight

    mu_x = weighted_avg_pool3x3(x)
    mu_y = weighted_avg_pool3x3(y)
    sigma_x = weighted_avg_pool3x3(x**2) - mu_x**2
    sigma_y = weighted_avg_pool3x3(y**2) - mu_y**2
    sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
    if c1 == float('inf'):
        ssim_n = (2 * sigma_xy + c2)
        ssim_d = (sigma_x + sigma_y + c2)
    elif c2 == float('inf'):
        ssim_n = 2 * mu_x * mu_y + c1
        ssim_d = mu_x**2 + mu_y**2 + c1
    else:
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    result = ssim_n / ssim_d
    return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight


def _avg_pool3x3(x):
    return torch.nn.AvgPool2d(kernel_size=3, stride=1)(x)


def _weighted_average(x, w, epsilon=1.0):
    weighted_sum = torch.sum(x * w, dim=(1, 2), keepdim=True)
    sum_of_weights = torch.sum(w, dim=(1, 2), keepdim=True)
    return weighted_sum / (sum_of_weights + epsilon)


def _expand_dims_twice(x, dim):
    return torch.unsqueeze(torch.unsqueeze(x, dim), dim)