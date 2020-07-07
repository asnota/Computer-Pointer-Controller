'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IECore, IENetwork
import numpy as np
import math
import cv2
import logging

class Model_GazeEstimation:
	'''
	Class for the Gaze Estimation Model.
	'''
	def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
		self.model_structure = model_path
		self.model_weights = model_path.replace('.xml', '.bin')
		self.device_name = device
		self.threshold = threshold
		self.logger = logging.getLogger('ge')
		self.model_name = 'Gaze Estimation Model'
		
		try:
			self.core = IECore()
			self.model = IENetwork(self.model_structure, self.model_weights)
		except Exception as e:
			self.logger.error("Error occured while initializing" + str(self.model_name) + str(e))
			raise ValueError("Could not initialize the network. Have you enterred the correct model path?")
		
		self.input_name = [i for i in self.model.inputs.keys()]
		self.input_shape = self.model.inputs[self.input_name[1]].shape
		self.output_name = [o for o in self.model.outputs.keys()]		
		self.network = None

	def load_model(self):
		try:
			self.network = self.core.load_network(network=self.model, device_name=self.device_name, num_requests=1)
		except Exception as e:
			self.logger.error("Error occured in load_model() method of" + str(self.model_name) + str(e))

	def predict(self, left_eye_image, right_eye_image, pose_output, request_id=0):
		try:
			left_eye_image = self.preprocess_input(left_eye_image)
		except Exception as e:
			self.logger.error("Error occured in predict() method of " + str(self.model_name) + str(e))
		try:
			right_eye_image = self.preprocess_input(right_eye_image)
			self.network.start_async(request_id, inputs={'left_eye_image': left_eye_image,
													 'right_eye_image': right_eye_image,
													 'head_pose_angles': pose_output})
			if self.wait() == 0:
				outputs = self.network.requests[0].outputs
				mouse_coords, gaze_vector = self.preprocess_output(outputs, pose_output)
		except Exception as e:
			self.logger.error("Error occured in predict() method of " + str(self.model_name) + str(e))
		return mouse_coords, gaze_vector

	def preprocess_input(self, image):
		print("Received image: ", image.shape)
		try:
			print(self.input_shape[3])
			print(self.input_shape[2])
			print(image.shape)
			image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]), cv2.INTER_AREA)				
		except Exception as e:
			self.logger.error("Error occured while image resize in preprocess_input() method of " + str(self.model_name) + str(e))
		try:
			image = image.transpose((2, 0, 1))
			print("Image shape after transpose: ", image.shape)
		except Exception as e:
			self.logger.error("Error occured while image transpose in preprocess_input() method of " + str(self.model_name) + str(e))
		try:
			image = image.reshape(1, *image.shape)
			print("Image shape after reshape: ", image.shape)
		except Exception as e:
			self.logger.error("Error occured while image reshape in preprocess_input() method of " + str(self.model_name) + str(e))
		
		return image

	def preprocess_output(self, outputs, pose_output):
		gaze_vector = outputs[self.output_name[0]][0]
		mouse_coords = (0, 0)
		try:
			angle_r_fc = pose_output[2]
			sin_r = math.sin(angle_r_fc * math.pi / 180.0)
			cos_r = math.cos(angle_r_fc * math.pi / 180.0)
			x = gaze_vector[0] * cos_r + gaze_vector[1] * sin_r
			y = -gaze_vector[0] * sin_r + gaze_vector[1] * cos_r
			mouse_coords = (x, y)
		except Exception as e:
			self.logger.error("Error occured in preprocess_output() method of " + str(self.model_name) + str(e))
		return mouse_coords, gaze_vector
		
	def wait(self):
		status = self.network.requests[0].wait(-1)
		return status
