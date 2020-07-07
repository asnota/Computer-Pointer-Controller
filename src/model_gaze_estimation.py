'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IECore, IENetwork
import numpy as np
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
		
		self.input_name = next(iter(self.model.inputs))
		self.input_shape = self.model.inputs[self.input_name].shape
		self.output_name = next(iter(self.model.outputs))
		self.output_shape = self.model.outputs[self.output_name].shape
		self.network = None

	def load_model(self):
		try:
			self.network = self.core.load_network(network=self.model, device_name=self.device_name, num_requests=1)
		except Exception as e:
			self.logger.error("Error While Loading"+str(self.model_name)+str(e))

	def predict(self, image):
		'''
		TODO: You will need to complete this method.
		This method is meant for running predictions on the input image.
		'''
		raise NotImplementedError

	def check_model(self):
		raise NotImplementedError

	def preprocess_input(self, image):
		'''
		Before feeding the data into the model for inference,
		you might have to preprocess it. This function is where you can do that.
		'''
		raise NotImplementedError

	def preprocess_output(self, outputs):
		'''
		Before feeding the output of this model to the next model,
		you might have to preprocess the output. This function is where you can do that.
		'''
		raise NotImplementedError
