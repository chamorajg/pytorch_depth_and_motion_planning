import json
import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

import intrinsics_utils
from depth_prediction_net import DispNetS
from object_motion_net import MotionVectorNet
from transform_utils import matrix_from_angles, angles_from_matrix

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--gpus', type=int, default=0,
                            help='how many gpus')
parser.add_argument('--model_path', type=str, default='./checkpoints/model.ckpt',
                            help='define the model path to test on the videos.')
parser.add_argument('--intrinsics', dest='intrinsics', action='store_true',
                            help='to use predefined intrinsics matrix in intrinsics.txt file.')
args = parser.parse_args()

def infer_ego_motion(rot, trans):
    """
        Infer ego motion (pose) using rot and trans matrix.
        Args:
            rot : rotation matrix.
            trans : translational matrix.
        Returns :
            avg_rot : rotation matrix for trajectory in world co-ordinates system.
            avg_trans : translation matrix for trajectory in world co-ordinates system.
    """
    rot12, rot21 = rot
    rot12 = matrix_from_angles(rot12)
    rot21 = matrix_from_angles(rot21)
    trans12, trans21 = trans

    avg_rot = 0.5 * (torch.linalg.inv(rot21) + rot12)
    avg_trans = 0.5 * (-torch.squeeze(
        torch.matmul(rot12, torch.unsqueeze(trans21, -1)), dim=-1) + trans12)
    return avg_rot, avg_trans

def odometry():
    """
        Preprocess frames.
        Resize the frames to (128, 416) size and then convert it into Tensor
        Args:
            frames : list of frames passed.s
        Returns:
            inference_stack : stack of preprocessed sequential images
    """
    
    # Resize and Transform.
    frames_list = json.load(open('./data/test.json'))
    transform=transforms.Compose([
                                transforms.Resize(size=(128,416)),
                                transforms.ToTensor(),
                            ])
    trajectory, positions = [], []
    position = np.zeros(3)
    orientation = np.eye(3)
    
    # Model Architecture
    if args.gpus != -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    intrinsics_mat=None
    if args.intrinsics:
        intrinsics_mat = np.loadtxt('./intrinsics.txt', delimiter=',')
        intrinsics_mat = intrinsics_mat.reshape(3, 3)

    depth_net = DispNetS()
    object_motion_net = MotionVectorNet(auto_mask=True, intrinsics=args.intrinsics, intrinsics_mat=intrinsics_mat)
    # Load Model
    model = torch.load(args.model_path, map_location=device)["state_dict"]
    depth_model = { k.replace("depth_net.", "") : v for k, v in model.items() if "depth" in k}
    depth_net.load_state_dict(depth_model)
    object_model = { k.replace("object_motion_net.", "") : v for k, v in model.items() if "object" in k}
    object_motion_net.load_state_dict(object_model)
    depth_net.eval()
    object_motion_net.eval()

    for i in range(len(frames_list)):
        sample_a = Image.open(frames_list[str(i)][0])
        sample_b = Image.open(frames_list[str(i)][1])
        sample_a = transform(sample_a)
        sample_b = transform(sample_b)
        endpoints = {}
        rgb_seq_images = [sample_a.unsqueeze(0), sample_b.unsqueeze(0)]
        rgb_images = torch.cat((rgb_seq_images[0], rgb_seq_images[1]), dim=0)
        
        depth_images = depth_net(rgb_images)
        depth_seq_images = torch.split(depth_images, depth_images.shape[0] // 2, dim=0)
        
        endpoints['predicted_depth'] = depth_seq_images
        endpoints['rgb'] = rgb_seq_images
        motion_features = [     
                torch.cat((endpoints['rgb'][0], 
                        endpoints['predicted_depth'][0]), dim=1),
                torch.cat((endpoints['rgb'][1], 
                        endpoints['predicted_depth'][1]), dim=1)]
        motion_features_stack = torch.cat(motion_features, dim=0)
        flipped_motion_features_stack = torch.cat(motion_features[::-1], dim=0)
        pairs = torch.cat([motion_features_stack, 
                        flipped_motion_features_stack], dim=1)
        
        rot, trans, residual_translation, intrinsics_mat = \
                                        object_motion_net(pairs)
        endpoints['residual_translation'] = torch.split(residual_translation, 
                                residual_translation.shape[0] // 2, dim=0)
        endpoints['background_translation'] = torch.split(trans, 
                                            trans.shape[0] // 2, dim=0)
        endpoints['rotation'] = torch.split(rot, rot.shape[0] // 2, dim=0)
        intrinsics_mat = 0.5 * sum(
            torch.split(intrinsics_mat, 
                            intrinsics_mat.shape[0] // 2, dim=0))
        endpoints['intrinsics_mat'] = [intrinsics_mat] * 2
        endpoints['intrinsics_mat_inv'] = [
        intrinsics_utils.invert_intrinsics_matrix(intrinsics_mat)] * 2

        rot, trans = infer_ego_motion(endpoints['rotation'], 
                    endpoints['background_translation'])
        rot_angles = angles_from_matrix(rot).detach().cpu().numpy()
        rot, trans = rot.detach().cpu().numpy(), trans.detach().cpu().numpy()
        orientation = np.dot(orientation, rot[0])
        trajectory.append(np.concatenate((np.concatenate((orientation, trans.T), axis=1), [[0, 0, 0, 1]]), axis=0))
        position += np.dot(orientation, trans[0])
        positions.append(position)
    trajectory = np.vstack(trajectory) # Trajectories - 4x4 Pose matrix will be stored in [(N-1)*4,4] vector in trajectory.txt
    positions = np.array(positions) # Positions - 1x3 will be stored as [(N-1),3] vector in positions.txt
    np.savetxt('./trajectory.txt', trajectory)
    np.savetxt('./positions.txt', positions)


if __name__ == "__main__":
    odometry()