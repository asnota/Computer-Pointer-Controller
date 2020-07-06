import cv2
import os
import logging
import time
import numpy as np
from input_feeder import InputFeeder
from mouse_controller import MouseController
from model_face_detection import Model_FaceDetection
from model_landmark_detection import Model_LandmarkDetection
from model_head_pose_estimation import Model_HeadPoseEstimation
from model_gaze_estimation import Model_GazeEstimation
from argparse import ArgumentParser



def build_argparser():
    """
    parse commandline argument
    return ArgumentParser object
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--faceDetectionModel", type=str, required=True,
                        help="Specify path of xml file of face detection model")

    parser.add_argument("-lr", "--landmarkRegressionModel", type=str, required=True,
                        help="Specify path of xml file of landmark regression model")

    parser.add_argument("-hp", "--headPoseEstimationModel", type=str, required=True,
                        help="Specify path of xml file of Head Pose Estimation model")

    parser.add_argument("-ge", "--gazeEstimationModel", type=str, required=True,
                        help="Specify path of xml file of Gaze Estimation model")

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Specify path of input Video file or cam for webcam")
    
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Specify probability threshold for face detection model")

    parser.add_argument("-d", "--device", required=False, type=str, default='CPU',
                        help="Specify Device for inference"
                             "It can be CPU, GPU, FPGU, MYRID")
    parser.add_argument("-o", '--output_path', default='/results/', type=str)
    return parser
	
def main():
    args = build_argparser().parse_args()
    logger = logging.getLogger('main')


if __name__ == '__main__':
    main()