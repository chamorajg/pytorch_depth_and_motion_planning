from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_

def create_scales(constraint_minimum):
    initialise = 0.01
    def constraint(x):
        return float(nn.ReLU(inplace=False)(torch.tensor(x - constraint_minimum).type(torch.float))) + constraint_minimum
    rot_scale = constraint(initialise)
    trans_scale = constraint(initialise)
    return rot_scale, trans_scale

class MotionVectorNet(nn.Module):

    def __init__(self, auto_mask=False, intrinsics=False, intrinsics_mat=None):
        super(MotionVectorNet, self).__init__()
        self.C = 8
        self.conv1 = nn.Conv2d(self.C, 16, kernel_size=3, stride=2, padding=1) # -> [B, 4, ]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]
        self.base_model = nn.Sequential( *self.layers
        )
        self._background_motion = nn.Conv2d(1024, 6, kernel_size=1, stride=1, padding=0)
        self._residual_translation = nn.Conv2d(6, 3, kernel_size=1, stride=1, padding=0)
        self.outputs = []
        self._refine_conv7 = self._refine_motion_field_conv7()
        self._refine_conv7_add = self._refine_motion_field_conv7_add()
        self._refine_conv6 = self._refine_motion_field_conv6()
        self._refine_conv6_add = self._refine_motion_field_conv6_add()
        self._refine_conv5 = self._refine_motion_field_conv5()
        self._refine_conv5_add = self._refine_motion_field_conv5_add()
        self._refine_conv4 = self._refine_motion_field_conv4()
        self._refine_conv4_add = self._refine_motion_field_conv4_add()
        self._refine_conv3 = self._refine_motion_field_conv3()
        self._refine_conv3_add = self._refine_motion_field_conv3_add()
        self._refine_conv2 = self._refine_motion_field_conv2()
        self._refine_conv2_add = self._refine_motion_field_conv2_add()
        self._refine_conv1 = self._refine_motion_field_conv1()
        self._refine_conv1_add = self._refine_motion_field_conv1_add()
        self._refine_conv = self._refine_motion_field_conv()
        self._refine_conv_add = self._refine_motion_field_conv_add()
        self.auto_mask = auto_mask
        self.intrinsics = intrinsics
        if self.intrinsics:
            self.intrinsics_mat = torch.from_numpy(intrinsics_mat).float()
            self.intrinsics_mat.unsqueeze(0)
        else:
            self.intrinsics_layer = nn.Sequential(  nn.Conv2d(1024, 2, kernel_size=1, stride=1),
                                                    nn.Softplus())
            self.intrinsics_layer_offset = nn.Conv2d(1024, 2, kernel_size=1, stride=1)
    
    def print_forward(self, x):
        self.outputs.append(x.detach())
        for l in self.layers:
            x = l(x)
            self.outputs.append(x.detach())
            # print(x.shape)
        return x
    
    def _concat_outputs(self, a, b):
        return torch.cat((a, b), dim=1)

    def _upsample_and_concat(self, motion_field):
        conv_size = (self.outputs[-1].shape[2], self.outputs[-1].shape[3])
        upsampled_motion_field = F.interpolate(motion_field, size=conv_size, mode='bilinear', align_corners=False)
        conv_input = torch.cat((upsampled_motion_field, self.outputs[-1].to(device=motion_field.device)), dim=1)
        self.refine_kernel = max(4, list(self.outputs[-1].shape)[1])
        # print(self.refine_kernel)
        self.outputs.pop()
        return conv_input, upsampled_motion_field
    
    def padding(self, x):
        in_height, in_width = x.shape[2], x.shape[3]
        out_height = ceil(float(in_height) / float(1))
        out_width = ceil(float(in_width) / float(1))

    def _refine_motion_field_conv7(self):
        return nn.Sequential(nn.Conv2d(1027, 1024, kernel_size=3, stride=1, padding=1),
                        nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1))
    
    def _refine_motion_field_conv7_add(self):
        return nn.Conv2d(2048, 3, kernel_size=1, stride=1)
    
    def _refine_motion_field_conv6(self):
        return nn.Sequential(nn.Conv2d(515, 512, kernel_size=3, stride=1, padding=1),
                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
    
    def _refine_motion_field_conv6_add(self):
        return nn.Conv2d(1024, 3, kernel_size=1, stride=1)
    
    def _refine_motion_field_conv5(self):
        return nn.Sequential(nn.Conv2d(259, 256, kernel_size=3, stride=1, padding=1),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
    
    def _refine_motion_field_conv5_add(self):
        return nn.Conv2d(512, 3, kernel_size=1, stride=1)

    def _refine_motion_field_conv4(self):
        return nn.Sequential(nn.Conv2d(131, 128, kernel_size=3, stride=1, padding=1),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
    
    def _refine_motion_field_conv4_add(self):
        return nn.Conv2d(256, 3, kernel_size=1, stride=1)
    
    def _refine_motion_field_conv3(self):
        return nn.Sequential(nn.Conv2d(67, 64, kernel_size=3, stride=1, padding=1),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
    
    def _refine_motion_field_conv3_add(self):
        return nn.Conv2d(128, 3, kernel_size=1, stride=1)
    
    def _refine_motion_field_conv2(self):
        return nn.Sequential(nn.Conv2d(35, 32, kernel_size=3, stride=1, padding=1),
                        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
    
    def _refine_motion_field_conv2_add(self):
        return nn.Conv2d(64, 3, kernel_size=1, stride=1)

    def _refine_motion_field_conv1(self):
        return nn.Sequential(nn.Conv2d(19, 16, kernel_size=3, stride=1, padding=1),
                        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1))
    
    def _refine_motion_field_conv1_add(self):
        return nn.Conv2d(32, 3, kernel_size=1, stride=1)
    
    def _refine_motion_field_conv(self):
        return nn.Sequential(nn.Conv2d(self.C+3, self.C, kernel_size=3, stride=1, padding=1),
                        nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1))
    
    def _refine_motion_field_conv_add(self):
        return nn.Conv2d(2*self.C, 3, kernel_size=1, stride=1)

    def _refine_motion_field(self, x):
        x_7, upsampled_motion_field = self._upsample_and_concat(x)
        a, b = self._refine_conv7[:-1](x_7), self._refine_conv7(x_7)
        conv_output = self._concat_outputs(a, b)
        x_7a = upsampled_motion_field + self._refine_conv7_add(conv_output)
        x_6, upsampled_motion_field = self._upsample_and_concat(x_7a)
        a, b = self._refine_conv6[:-1](x_6), self._refine_conv6(x_6)
        conv_output = self._concat_outputs(a, b)
        x_6a = upsampled_motion_field + self._refine_conv6_add(conv_output)
        x_5, upsampled_motion_field = self._upsample_and_concat(x_6a)
        a, b = self._refine_conv5[:-1](x_5), self._refine_conv5(x_5)
        conv_output = self._concat_outputs(a, b)
        x_5a = upsampled_motion_field + self._refine_conv5_add(conv_output)
        x_4, upsampled_motion_field = self._upsample_and_concat(x_5a)
        a, b = self._refine_conv4[:-1](x_4), self._refine_conv4(x_4)
        conv_output = self._concat_outputs(a, b)
        x_4a = upsampled_motion_field + self._refine_conv4_add(conv_output)
        x_3, upsampled_motion_field = self._upsample_and_concat(x_4a)
        a, b = self._refine_conv3[:-1](x_3), self._refine_conv3(x_3)
        conv_output = self._concat_outputs(a, b)
        x_3a = upsampled_motion_field + self._refine_conv3_add(conv_output)
        x_2, upsampled_motion_field = self._upsample_and_concat(x_3a)
        a, b = self._refine_conv2[:-1](x_2), self._refine_conv2(x_2)
        conv_output = self._concat_outputs(a, b)
        x_2a = upsampled_motion_field + self._refine_conv2_add(conv_output)
        x_1, upsampled_motion_field = self._upsample_and_concat(x_2a)
        a, b = self._refine_conv1[:-1](x_1), self._refine_conv1(x_1)
        conv_output = self._concat_outputs(a, b)
        x_1a = upsampled_motion_field + self._refine_conv1_add(conv_output)
        x, upsampled_motion_field = self._upsample_and_concat(x_1a)
        a, b = self._refine_conv[:-1](x), self._refine_conv(x)
        conv_output = self._concat_outputs(a, b)
        x_a = upsampled_motion_field + self._refine_conv_add(conv_output)
        return x_a
    
    def _mask(self, x):
        sq_x = torch.sqrt(torch.sum(x**2,
                                 dim=1, keepdim=True))
        mean_sq_x = torch.mean(sq_x, dim=(0, 2, 3))
        mask_x = (sq_x > mean_sq_x).type(x.dtype)
        x = x * mask_x
        return x

    def _intrinsic_layer(self, x, h, w):
        batch_size = x.shape[0]
        offsets = self.intrinsics_layer_offset(x)
        focal_lengths = self.intrinsics_layer(x)
        focal_lengths = focal_lengths.squeeze(2).squeeze(2) + 0.5
        focal_lengths = focal_lengths * torch.tensor([[w, h]], 
                        dtype=x.dtype, device=x.device)
        offsets = offsets.squeeze(2).squeeze(2) + 0.5
        offsets = offsets * torch.tensor([[w, h]], 
                        dtype=x.dtype, device=x.device)
        foci = torch.diagflat(focal_lengths[0]).unsqueeze(0)
        for b in range(1, batch_size):
            foci = torch.cat((foci, torch.diagflat(focal_lengths[b]).unsqueeze(0)), dim=0)
        intrinsic_mat = torch.cat([foci, torch.unsqueeze(offsets, -1)], dim=2)
        last_row = torch.tensor([[[0.0, 0.0, 1.0]]]).repeat(batch_size, 1, 1).to(
                            device=x.device)
        intrinsic_mat = torch.cat([intrinsic_mat, last_row], dim=1)
        return intrinsic_mat
    
    def forward(self, x):
        x = self.print_forward(x)
        batch_size = x.shape[0]
        bottleneck = torch.mean(x, dim=(2,3), keepdim=True)
        background_motion = self._background_motion(bottleneck)
        rotation = background_motion[:, :3, 0, 0].clone()
        background_translation = background_motion[:, 3:, :, :].clone()
        residual_translation = self._residual_translation(background_motion)
        residual_translation = self._refine_motion_field(residual_translation)
        rot_scale, trans_scale = create_scales(0.001)
        background_translation *= trans_scale
        residual_translation *= trans_scale
        rotation *= rot_scale
        if self.auto_mask:
            residual_translation = self._mask(residual_translation)
        image_height, image_width = x.shape[2], x.shape[3]
        if self.intrinsics:
            intrinsic_mat = self.intrinsics_mat.repeat(batch_size, 1, 1).to(x.device)
        else:
            intrinsic_mat = self._intrinsic_layer(bottleneck, image_height, image_width)
        return (rotation, background_translation.reshape(-1,3), residual_translation.clone().reshape(-1,128,416,3), intrinsic_mat)

if __name__=="__main__":
    mvn = MotionVectorNet(auto_mask = True).to(device='cuda')
    x = torch.randn((10,8,128,416))
    x = x.to(device='cuda')
    o = mvn(x)
    print(o[0].shape, o[1].shape, o[2].shape, o[3].shape)