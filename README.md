### Pytorch Implementation of Unsupervised Depth and Motion Learning.

This project is an unofficial pytorch implementation of https://github.com/google-research/google-research/tree/master/depth_and_motion_learning
This project is based on the paper https://arxiv.org/abs/2010.16404

# Installation Instructions
pip install -r requirements.txt

# Loading the data 
the image sequences has to be loaded in the format given below:
    For training:
        {"0": ["./images/video/frame000.png", "./images/video/frame001.png"], "1": ["./images/video/frame001.png", "./images/video/frame002.png"]}
        and stored in data/train.json
    For Validating:
        {"0": ["./images/video/frame000.png", "./images/video/frame001.png"], "1": ["./images/video/frame001.png", "./images/video/frame002.png"]}
        and stored in data/valid.json
    For testing:
        {"0": ["./images/video/frame000.png", "./images/video/frame001.png"], "1": ["./images/video/frame001.png", "./images/video/frame002.png"]}
        and stored in data/test.json

# Specifying given intrinsics:
    The intrinsics must be written in the list format separated by commas (Eg - fx,0,cx,0,fy,cy,0,0,1) and saved in intrinsics.txt

# Training the Model :
    By default the model uses certain parameters specified in train.py __init__() function which can be changed according to requirements.
    By default model predicts the intrinsics of the video given.
    To use the predefined intrinsics intrinsics must be definied in the intrinsics.txt file.

# Testing the Model :
    To test the model, specify the model path and the test.json which contains the images.
    4x4 Pose Matrix along with the 3 translatory motion (x,y,z) co-ordinates will be saved to trajectory.txt (Pose) and positions.txt

# ToDo:
    1.) To train and validate intrinsics per video while training, validating and testing.
    2.) Add pretrained models on KITTI Dataset and Waymo Open Dataset.

# Citation
If you use any part of this code in your research, please cite this paper:

@article{li2020unsupervised,
  title={Unsupervised Monocular Depth Learning in Dynamic Scenes},
  author={Li, Hanhan and Gordon, Ariel and Zhao, Hang and Casser, Vincent and Angelova, Anelia},
  journal={arXiv preprint arXiv:2010.16404},
  year={2020}
}

# Contributing
If you find a bug, create a GitHub issue, or even better, submit a pull request. 
Similarly, if you have questions, simply post them as GitHub issues.