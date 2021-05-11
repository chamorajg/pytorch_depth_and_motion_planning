from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import torch

def joint_bilateral_smoothing(smoothed, reference):
    """Computes edge-aware smoothness loss.
    Args:
    smoothed: A torch.Tensor of shape [B, C1, H, W] to be smoothed.
    reference: A torch.Tensor of the shape [B, C2, H, W]. Wherever `reference` has
        more spatial variation, the strength of the smoothing of `smoothed` will
        be weaker.
    Returns:
    A scalar torch.Tensor containing the regularization, to be added to the
    training loss.
    """
    smoothed_dx = _gradient_x(smoothed)
    smoothed_dy = _gradient_y(smoothed)
    ref_dx = _gradient_x(reference)
    ref_dy = _gradient_y(reference)
    weights_x = torch.exp(-torch.mean(torch.abs(ref_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(ref_dy), dim=1, keepdim=True))
    smoothness_x = smoothed_dx * weights_x
    smoothness_y = smoothed_dy * weights_y
    return torch.mean(abs(smoothness_x)) + torch.mean(abs(smoothness_y))


def normalize_motion_map(res_motion_map, motion_map):
    """Normalizes a residual motion map by the motion map's norm."""
    norm = torch.mean(
        torch.square(motion_map), dim=[1, 2, 3], keepdim=True) * 3.0
    return res_motion_map / torch.sqrt(norm + 1e-12)


def l1smoothness(tensor, wrap_around=True):
    """Calculates L1 (total variation) smoothness loss of a tensor.
    Args:
    tensor: A tensor to be smoothed, of shape [B, C, H, W].
    wrap_around: True to wrap around the last pixels to the first.
    Returns:
    A scalar torch.Tensor, The total variation loss.
    """
    tensor_dx = tensor - torch.roll(tensor, 2, 2)
    tensor_dy = tensor - torch.roll(tensor, 2, 3)
    # We optionally wrap around in order to impose continuity across the
    # boundary. The motivation is that there is some ambiguity between rotation
    # and spatial gradients of translation maps. We would like to discourage
    # spatial gradients of the translation field, and to absorb sich gradients
    # into the rotation as much as possible. This is why we impose continuity
    # across the spatial boundary.
    if not wrap_around:
        tensor_dx = tensor_dx[:, :, 1:, 1:]
        tensor_dy = tensor_dy[:, :, 1:, 1:]
    return torch.mean(
        torch.sqrt(1e-24 + torch.square(tensor_dx) + torch.square(tensor_dy)))


def sqrt_sparsity(motion_map):
    """A regularizer that encourages sparsity.
    This regularizer penalizes nonzero values. Close to zero it behaves like an L1
    regularizer, and far away from zero its strength decreases. The scale that
    distinguishes "close" from "far" is the mean value of the absolute of
    `motion_map`.
    Args:
        motion_map: A torch.Tensor of shape [B, C, H, W]
    Returns:
        A scalar torch.Tensor, the regularizer to be added to the training loss.
    """
    tensor_abs = torch.abs(motion_map)
    mean = torch.mean(tensor_abs, dim=(2, 3), keepdim=True).detach()
    # We used L0.5 norm here because it's more sparsity encouraging than L1.
    # The coefficients are designed in a way that the norm asymptotes to L1 in
    # the small value limit.
    return torch.mean(2 * mean * torch.sqrt(tensor_abs / (mean + 1e-24) + 1))


def _gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def _gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]