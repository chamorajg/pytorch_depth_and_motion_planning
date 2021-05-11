import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import intrinsics_utils
import consistency_losses
import regularizers
import transform_depth_map

def _get_intrinsics_mat_pyramid(intrinsics_mat, num_scales):
    """Returns multiple intrinsic matrices for different scales.

    Args:
        intrinsics_mat: <float32>[B, 3, 3] tensor containing the intrinsics matrix
        at the original scale.
        num_scales: integer indicating *total* number of matrices to return.  If
        `num_scales` is 1, the function just returns the input matrix in a list.

    Returns:
        List containing `num_scales` intrinsics matrices, each with shape
        <float32>[B, 3, 3].  The first element in the list is the input
        intrinsics matrix and the last element is the intrinsics matrix for the
        coarsest scale.
    """
    # intrinsics_mat: [B, 3, 3]
    intrinsics_mat_pyramid = [intrinsics_mat]
    # Scale the intrinsics accordingly for each scale.
    for s in range(1, num_scales):
        fx = intrinsics_mat[:, 0, 0] / 2**s
        fy = intrinsics_mat[:, 1, 1] / 2**s
        cx = intrinsics_mat[:, 0, 2] / 2**s
        cy = intrinsics_mat[:, 1, 2] / 2**s
        intrinsics_mat_pyramid.append(_make_intrinsics_matrix(fx, fy, cx, cy))
    return intrinsics_mat_pyramid


def _make_intrinsics_matrix(fx, fy, cx, cy):
    """Constructs a batch of intrinsics matrices given arguments..

    Args:
        fx: <float32>[B] tensor containing horizontal focal length.
        fy: <float32>[B] tensor containing vertical focal length.
        cx: <float32>[B] tensor containing horizontal principal offset.
        cy: <float32>[B] tensor containing vertical principal offset.

    Returns:
        <float32>[B, 3, 3] tensor containing batch of intrinsics matrices.
    """
    # fx, fy, cx, cy: [B]
    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)
    r1 = torch.stack([fx, zeros, cx], dim=-1)
    r2 = torch.stack([zeros, fy, cy], dim=-1)
    r3 = torch.stack([zeros, zeros, ones], dim=-1)
    intrinsics = torch.stack([r1, r2, r3], dim=1)
    return intrinsics


def _min_pool2d(input_, ksize, strides, padding):
    return -torch.nn.MaxPool2d(ksize, strides, padding=None)(-input_)


def _get_pyramid(img, num_scales, pooling_fn=torch.nn.AvgPool2d):
    """Generates a pyramid from the input image/tensor at different scales.

    This function behaves similarly to `tfg.image.pyramid.split()`.  Instead of
    using an image resize operation, it uses average pooling to give each
    input pixel equal weight in constructing coarser scales.

    Args:
        img: [B, height, width, C] tensor, where B stands for batch size and C
        stands for number of channels.
        num_scales: integer indicating *total* number of scales to return.  If
        `num_scales` is 1, the function just returns the input image in a list.
        pooling_fn: A callable with tf.nn.avg_pool2d's signature, to be used for
        pooling `img` across scales.

    Returns:
        List containing `num_scales` tensors with shapes
        [B, height / 2^s, width / 2^s, C] where s is in [0, num_scales - 1].  The
        first element in the list is the input image and the last element is the
        resized input corresponding to the coarsest scale.
    """
    pyramid = [img]
    for _ in range(1, num_scales):
        # Scale image stack.
        last_img = pyramid[-1]
        scaled_img = pooling_fn(2, 2, padding=None)(last_img)
        pyramid.append(scaled_img)
    return pyramid

class DMPLoss(nn.Module):

    def __init__(self, default_weights):
        super(DMPLoss, self).__init__()
        self.default_weights = default_weights
        self.default_params = {
                            'target_depth_stop_gradient': True,
                            'scale_normalization': False,
                            'num_scales': 1,
            }
        self._output_endpoints = {}

    def _reinitialise_losses(self, device):
        _losses = {k:torch.tensor(0.0).to(device) for k in self.default_weights.keys()}
        return _losses
        
    def forward(self, endpoints):
        rgb_stack_ = torch.cat(endpoints['rgb'], dim=0)
        flipped_rgb_stack_ = torch.cat(endpoints['rgb'][::-1], dim=0)
        predicted_depth_stack_ = torch.cat(
                    endpoints['predicted_depth'], dim=0)
        flipped_predicted_depth_stack_ = torch.cat(
                endpoints['predicted_depth'][::-1], dim=0)
        residual_translation_ = torch.cat(
            endpoints['residual_translation'], dim=0)
        flipped_residual_translation_ = torch.cat(
            endpoints['residual_translation'][::-1], dim=0)
        intrinsics_mat_ = torch.cat(endpoints['intrinsics_mat'], dim=0)

        _losses = self._reinitialise_losses(rgb_stack_.device)
        

        # Create pyramids from each stack to support multi-scale training.
        num_scales = self.default_params['num_scales']
        rgb_pyramid = _get_pyramid(rgb_stack_, num_scales=num_scales)
        flipped_rgb_pyramid = _get_pyramid(
            flipped_rgb_stack_, num_scales=num_scales)
        predicted_depth_pyramid = _get_pyramid(
            predicted_depth_stack_, num_scales=num_scales)
        flipped_predicted_depth_pyramid = _get_pyramid(
            flipped_predicted_depth_stack_, num_scales=num_scales)
        residual_translation_pyramid = _get_pyramid(
            residual_translation_, num_scales=num_scales)
        flipped_residual_translation_pyramid = _get_pyramid(
            flipped_residual_translation_, num_scales=num_scales)
        intrinsics_mat_pyramid = _get_intrinsics_mat_pyramid(
            intrinsics_mat_, num_scales=num_scales)

        validity_mask_ = endpoints.get('validity_mask')
        if validity_mask_ is not None:
            validity_mask_ = torch.cat(validity_mask_, dim=0)
            validity_mask_pyramid = _get_pyramid(
                validity_mask_, num_scales, _min_pool2d)
        else:
            validity_mask_pyramid = [None] * num_scales

        for s in reversed(range(self.default_params['num_scales'])):
            # Weight applied to all losses at this scale.
            scale_w = 1.0 / 2**s

            rgb_stack = rgb_pyramid[s]
            predicted_depth_stack = predicted_depth_pyramid[s]
            flipped_predicted_depth_stack = flipped_predicted_depth_pyramid[s]

            # In theory, the training losses should be agnostic to the global scale of
            # the predicted depth. However in reality second order effects can lead to
            # (https://en.wikipedia.org/wiki/Von_Neumann_stability_analysis) diverging
            # modes. For some reason this happens when training on TPU. Since the
            # scale is immaterial anyway, we normalize it out, and the training
            # stabilizes.
            #
            # Note that the depth supervision term, which is sensitive to the scale,
            # was applied before this normalization. Therefore the scale of the depth
            # is learned.
            mean_depth = torch.mean(predicted_depth_stack)

            # When training starts, the depth sometimes tends to collapse to a
            # constant value, which seems to be a fixed point where the trainig can
            # stuck. To discourage this collapse, we penalize the reciprocal of the
            # variance with a tiny weight. Note that the mean of predicted_depth is
            # one, hence we subtract 1.0.
            depth_var = torch.mean(
                torch.square(predicted_depth_stack / mean_depth - 1.0))

            if self.default_params['scale_normalization']:
                predicted_depth_stack /= mean_depth
                flipped_predicted_depth_stack /= mean_depth

            disp = 1.0 / predicted_depth_stack

            mean_disp = torch.mean(disp, dim=[1, 2, 3], keepdim=True)

            _losses['depth_variance'] = scale_w * 1.0 / depth_var

            _losses['depth_smoothing'] = _losses['depth_smoothing'] +(
                scale_w *
                regularizers.joint_bilateral_smoothing(disp / mean_disp, rgb_stack))

            self._output_endpoints['disparity'] = disp

            flipped_rgb_stack = flipped_rgb_pyramid[s]

            background_translation = torch.cat(
                endpoints['background_translation'], dim=0)
            flipped_background_translation = torch.cat(
                endpoints['background_translation'][::-1], dim=0)
            residual_translation = residual_translation_pyramid[s]
            flipped_residual_translation = flipped_residual_translation_pyramid[s]
            
            if self.default_params['scale_normalization']:
                background_translation /= mean_depth
                flipped_background_translation /= mean_depth
                residual_translation /= mean_depth
                flipped_residual_translation /= mean_depth
            
            translation = torch.add(residual_translation, background_translation.view(-1, 1, 1, 3))
            flipped_translation = (
                flipped_residual_translation + flipped_background_translation.view(-1, 1, 1,3))

            rotation = torch.cat(endpoints['rotation'], dim=0)
            flipped_rotation = torch.cat(endpoints['rotation'][::-1], dim=0)
            intrinsics_mat = intrinsics_mat_pyramid[s]
            intrinsics_mat_inv = intrinsics_utils.invert_intrinsics_matrix(
                intrinsics_mat)
            
            validity_mask = validity_mask_pyramid[s]

            transformed_depth = transform_depth_map.using_motion_vector(
                torch.squeeze(predicted_depth_stack, dim=1), translation, rotation,
                intrinsics_mat, intrinsics_mat_inv)
            flipped_predicted_depth_stack = torch.squeeze(
                flipped_predicted_depth_stack, dim=-1)
            if self.default_params['target_depth_stop_gradient']:
                flipped_predicted_depth_stack = flipped_predicted_depth_stack.detach()
            # The first and second halves of the batch not contain Frame1's and
            # Frame2's depths transformed onto Frame2 and Frame1 respectively. Te
            # demand consistency, we need to `flip` `predicted_depth` as well.
            loss_endpoints = (
                consistency_losses.rgbd_and_motion_consistency_loss(
                    transformed_depth,
                    rgb_stack,
                    flipped_predicted_depth_stack,
                    flipped_rgb_stack,
                    rotation,
                    translation,
                    flipped_rotation,
                    flipped_translation,
                    validity_mask=validity_mask))

            normalized_trans = regularizers.normalize_motion_map(
                residual_translation, translation)
            _losses['motion_smoothing'] = _losses['motion_smoothing'] + \
                scale_w * regularizers.l1smoothness(
                normalized_trans, self.default_weights['motion_drift'] == 0)
            _losses['motion_drift'] = _losses['motion_drift'] + \
                scale_w * regularizers.sqrt_sparsity(
                normalized_trans)
            _losses['depth_consistency'] = _losses['depth_consistency'] + (
                scale_w * loss_endpoints['depth_error'])
            _losses['rgb_consistency'] = _losses['rgb_consistency'] + \
                                    scale_w * loss_endpoints['rgb_error']
            _losses['ssim'] = _losses['ssim'] + \
                        scale_w * 0.5 * loss_endpoints['ssim_error']

            _losses['rotation_cycle_consistency'] = \
                _losses['rotation_cycle_consistency'] + (
                scale_w * loss_endpoints['rotation_error'])
            _losses['translation_cycle_consistency'] = \
                _losses['translation_cycle_consistency'] + (
                scale_w * loss_endpoints['translation_error'])

            self._output_endpoints['depth_proximity_weight'] = loss_endpoints[
                'depth_proximity_weight']
            self._output_endpoints['trans'] = translation
            self._output_endpoints['inv_trans'] = flipped_translation

        for k, w in self.default_weights.items():
            # multiply by 2 to match the scale of the old code.
            _losses[k] = _losses[k] * w * 2
        losses = sum(_losses.values())
        return losses