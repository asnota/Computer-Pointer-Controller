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
		self.logger = logging.getLogger('fd')
		self.model_name = 'Landmark Detection Model'
		
		try:
			self.core = IECore()
			self.model = IENetwork(self.model_structure, self.model_weights)
		except Exception as e:
			self.logger.error("Error While Initilizing" + str(self.model_name) + str(e))
			raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
		
		self.input_name = next(iter(self.model.inputs))
		self.input_shape = self.model.inputs[self.input_name].shape
		self.output_name = next(iter(self.model.outputs))
		self.output_shape = self.model.outputs[self.output_name].shape
		self.network = None

	def load_model(self):
		'''
		TODO: You will need to complete this method.
		This method is for loading the model to the device specified by the user.
		If your model requires any Plugins, this is where you can load them.
		'''
		raise NotImplementedError

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
