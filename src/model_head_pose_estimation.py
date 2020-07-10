'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IECore, IENetwork
import numpy as np
import cv2
import logging
from model import Model

class Model_HeadPoseEstimation(Model):
	'''
	Class for the Head Pose Estimation Model.
	'''
	def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
		
		Model.__init__(self, model_path, device, extensions, threshold)
		self.model_name = 'Head Pose Estimation Model'		
		self.input_name = next(iter(self.model.inputs))
		self.input_shape = self.model.inputs[self.input_name].shape
		self.output_name = next(iter(self.model.outputs))
		self.output_shape = self.model.outputs[self.output_name].shape
		self.network = None

	def predict(self, image, request_id=0):
		try:
			preprocessed_image = self.preprocess_input(image)
			self.network.start_async(request_id, inputs={self.input_name: preprocessed_image})
			if self.wait() == 0:
				global pose
				outputs = self.network.requests[0].outputs				
				#print("Outputs from head position keys", outputs.keys())				
				pose = self.preprocess_output(outputs)				
		except Exception as e:
			self.logger.error("Error occured in predict() method of " + str(self.model_name) + str(e))
		return pose

	def preprocess_input(self, image):
		try:
			image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
			image = image.transpose((2, 0, 1))
			image = image.reshape(1, *image.shape)
		except Exception as e:
			self.logger.error("Error occured in preprocess_input() method of " + str(self.model_name) + str(e))
		return image

	def preprocess_output(self, outputs):
		pose_output = []		
		try:
			pose_output.append(outputs['angle_y_fc'][0][0])				
			pose_output.append(outputs['angle_p_fc'][0][0])
			pose_output.append(outputs['angle_r_fc'][0][0])
		except Exception as e:
			self.logger.error("Error occured in preprocess_output() method of " + str(self.model_name) + str(e))
		return pose_output