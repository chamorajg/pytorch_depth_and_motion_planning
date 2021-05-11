import torch

def invert_intrinsics_matrix(intrinsics_mat):
    """Inverts an intrinsics matrix.
    Inverting matrices in not supported on TPU. The intrinsics matrix has however
    a closed form expression for its inverse, and this function invokes it.
    Args:
        intrinsics_mat: A tensor of shape [.... 3, 3], representing an intrinsics
        matrix `(in the last two dimensions).
    Returns:
        A tensor of the same shape containing the inverse of intrinsics_mat
    """
    intrinsics_mat = intrinsics_mat
    intrinsics_mat_cols = torch.unbind(intrinsics_mat, dim=-1)
    if len(intrinsics_mat_cols) != 3:
        raise ValueError('The last dimension of intrinsics_mat should be 3, not '
                    '%d.' % len(intrinsics_mat_cols))

    fx, _, _ = torch.unbind(intrinsics_mat_cols[0], dim=-1)
    _, fy, _ = torch.unbind(intrinsics_mat_cols[1], dim=-1)
    x0, y0, _ = torch.unbind(intrinsics_mat_cols[2], dim=-1)

    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)

    row1 = torch.stack([1.0 / fx, zeros, zeros], dim=-1)
    row2 = torch.stack([zeros, 1.0 / fy, zeros], dim=-1)
    row3 = torch.stack([-x0 / fx, -y0 / fy, ones], dim=-1)

    return torch.stack([row1, row2, row3], dim=-1)