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

FD_MODEL = "../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml"
LR_MODEL = "../intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"
HP_MODEL = "../intel/human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml"
GE_MODEL = "../intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml"
VIDEO_PATH = "demo.mp4"


def build_argparser():
    """
    parse commandline argument
    return ArgumentParser object
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--faceDetectionModel", type=str, required=False,
						default=FD_MODEL,
                        help="Specify path for xml file of the face detection model")

    parser.add_argument("-lr", "--landmarkRegressionModel", type=str, required=False,
						default=LR_MODEL,
                        help="Specify path for xml file of the landmark regression model")

    parser.add_argument("-hp", "--headPoseEstimationModel", type=str, required=False,
						default=HP_MODEL,
                        help="Specify path for xml file of the head pose estimation model")

    parser.add_argument("-ge", "--gazeEstimationModel", type=str, required=False,
						default=GE_MODEL,
                        help="Specify path for xml file of the gaze estimation model")

    parser.add_argument("-i", "--input", type=str, required=False,
						default=VIDEO_PATH,
                        help="Specify path for input video file or cam for webcam")
    
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Specify probability threshold for face detection model")

    parser.add_argument("-d", "--device", required=False, type=str, default='CPU',
                        help="Specify device for inference"
                             "It can be CPU, GPU, FPGU or MYRIAD")
    parser.add_argument("-o", '--output_path', default='/results/', type=str)
    return parser

def infer_on_stream(args):
	#Initialise variables with parsed arguments
	model_path_dict = {'FaceDetectionModel': args.faceDetectionModel,'LandmarkRegressionModel': args.landmarkRegressionModel,'HeadPoseEstimationModel': args.headPoseEstimationModel,'GazeEstimationModel': args.gazeEstimationModel}
	input_file = args.input
	device_name = args.device
	prob_threshold = args.prob_threshold
	output_path = args.output_path
	
	logger = logging.getLogger('infer_on_stream')
	
	#Check the video input
	if input_file.lower() == 'cam':
		feeder = InputFeeder(input_type='cam')
	else:
		print(input_file)
		if not os.path.isfile(input_file):
			logger.error("Unable to find specified video file")
			exit(1)
		feeder = InputFeeder(input_type='video', input_file=input_file)
	
	#Check the model input:
	for model_path in list(model_path_dict.values()):
		print(model_path)
		if not os.path.isfile(model_path):
			logger.error("Unable to find specified model file" + str(model_path))
			exit(1)
	
	#Initialize models
	face_detection_model = Model_FaceDetection(model_path_dict['FaceDetectionModel'], device_name, prob_threshold)
	landmark_detection_model = Model_LandmarkDetection(model_path_dict['LandmarkRegressionModel'], device_name, prob_threshold)
	head_pose_estimation_model = Model_HeadPoseEstimation(model_path_dict['HeadPoseEstimationModel'], device_name, prob_threshold)
	gaze_estimation_model = Model_GazeEstimation(model_path_dict['GazeEstimationModel'], device_name, prob_threshold)
	
	#Load models
	face_detection_model.load_model()
	landmark_detection_model.load_model()
	head_pose_estimation_model.load_model()
	gaze_estimation_model.load_model()

	#Load video input and precise video output
	mouse_controller = MouseController('medium', 'fast')
	feeder.load_data()	
	out_video = cv2.VideoWriter(os.path.join('output_video.mp4'), cv2.VideoWriter_fourcc('M','J','P','G'), int(feeder.get_fps()/10),
                                (1920, 1080), True)
	
	frame_count = 0
	for ret, frame in feeder.next_batch():
		global face_coords, cropped_image
		if not ret:
			break
		frame_count +=1
		key = cv2.waitKey(60)
		try:
			face_coords, cropped_image = face_detection_model.predict(frame)
		except Exception as e:
			logger.warning("Could not predict using model face_detection_model" + str(e) + " for frame " + str(frame_count))
		try:
			left_eye_image, right_eye_image, eye_coords = landmark_detection_model.predict(cropped_image)
		except Exception as e:
			logger.warning("Could not predict using landmark_detection_model" + str(e) + " for frame " + str(frame_count))
		try:
			pose_output = head_pose_estimation_model.predict(cropped_image)
		except Exception as e:
			logger.warning("Could not predict using head_pose_estimation_model" + str(e) + " for frame " + str(frame_count))
			continue
			
def main():
	args = build_argparser().parse_args()
	infer_on_stream(args)		

if __name__ == '__main__':
    main()