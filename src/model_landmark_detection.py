'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IECore, IENetwork
import numpy as np
import cv2
import logging

class Model_LandmarkDetection:
	'''
	Class for the Landmark Detection Model.
	'''
	def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
		self.model_structure = model_path
		self.model_weights = model_path.replace('.xml', '.bin')
		self.device_name = device
		self.threshold = threshold
		self.logger = logging.getLogger('ld')
		self.model_name = 'Landmark Detection Model'
		
		try:
			self.core = IECore()
			self.model = IENetwork(self.model_structure, self.model_weights)
		except Exception as e:
			self.logger.error("Error occured while initializing" + str(self.model_name) + str(e))
			raise ValueError("Could not initialize the network. Have you enterred the correct model path?")
		
		self.input_name = next(iter(self.model.inputs))
		self.input_shape = self.model.inputs[self.input_name].shape
		self.output_name = next(iter(self.model.outputs))
		self.output_shape = self.model.outputs[self.output_name].shape
		self.network = None

	def load_model(self):
		try:
			self.network = self.core.load_network(network=self.model, device_name=self.device_name, num_requests=1)
		except Exception as e:
			self.logger.error("Error occured in load_model() method of" + str(self.model_name)+str(e))

	def predict(self, image, request_id=0):
		left_eye_image, right_eye_image, eye_cords = [], [], []
		try:
			preprocessed_image = self.preprocess_input(image)
			self.network.start_async(request_id, inputs={self.input_name: preprocessed_image})
			if self.wait() == 0:
				outputs = self.network.requests[0].outputs[self.output_name]
				left_eye_image, right_eye_image, eye_cords = self.preprocess_output(outputs, image)
		except Exception as e:
			self.logger.error("Error occured in predict() method of " + str(self.model_name) + str(e))
		return left_eye_image, right_eye_image, eye_cords

	def preprocess_input(self, image):
		try:
			image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
			image = image.transpose((2, 0, 1))
			image = image.reshape(1, *image.shape)
		except Exception as e:
			self.logger.error("Error occured in preprocess_input() method of " + str(self.model_name) + str(e))
		return image		

	def preprocess_output(self, outputs, image):
		h = image.shape[0]
		w = image.shape[1]
		left_eye_image, right_eye_image, eye_cords = [], [], []
		try:
			outputs = outputs[0]

			left_eye_xmin = int(outputs[0][0][0] * w) - 10
			left_eye_ymin = int(outputs[1][0][0] * h) - 10
			right_eye_xmin = int(outputs[2][0][0] * w) - 10
			right_eye_ymin = int(outputs[3][0][0] * h) - 10

			left_eye_xmax = int(outputs[0][0][0] * w) + 10
			left_eye_ymax = int(outputs[1][0][0] * h) + 10
			right_eye_xmax = int(outputs[2][0][0] * w) + 10
			right_eye_ymax = int(outputs[3][0][0] * h) + 10

			left_eye_image = image[left_eye_ymin:left_eye_ymax, left_eye_xmin:left_eye_xmax]
			right_eye_image = image[right_eye_ymin:right_eye_ymax, right_eye_xmin:right_eye_xmax]

			eye_cords = [[left_eye_xmin, left_eye_ymin, left_eye_xmax, left_eye_ymax],
						 [right_eye_xmin, right_eye_ymin, right_eye_xmax, right_eye_ymax]]

		except Exception as e:
			self.logger.error("Error occured in preprocess_output() method of " + str(self.model_name) + str(e))
		return left_eye_image, right_eye_image, eye_cords
		
	def wait(self):
		status = self.network.requests[0].wait(-1)
		return status

