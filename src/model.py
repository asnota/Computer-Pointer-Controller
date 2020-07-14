'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IECore, IENetwork
import numpy as np
import cv2
import logging

class Model:	
	def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
		self.model_structure = model_path
		self.model_weights = model_path.replace('.xml', '.bin')
		self.device_name = device
		self.threshold = threshold
		self.logger = logging.getLogger('fd')
		self.model_name = 'Parent Model'
		
		try:
			self.core = IECore()
			self.model = IENetwork(self.model_structure, self.model_weights)			
		except Exception as e:
			self.logger.error("Error occured while initializing" + str(self.model_name) + str(e))
			raise ValueError("Could not initialize the network. Have you enterred the correct model path?")
			
		supported_layers = self.core.query_network(network=self.model, device_name="CPU")
		unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
		if len(unsupported_layers) != 0:
			raise ValueError("Unsupportedlayers found: {}".format(unsupported_layers))
			exit(1)		
		
		self.input_name = None
		self.input_shape = None
		self.output_name = None
		self.output_shape = None
		self.network = None

	def load_model(self):		
		try:
			self.network = self.core.load_network(network=self.model, device_name=self.device_name, num_requests=1)
		except Exception as e:
			self.logger.error("Error occured in load_model() method of " + str(self.model_name) + str(e))
			
	def predict(self, image, request_id=0):
		pass


	def preprocess_input(self, image):
		pass


	def preprocess_output(self, outputs, image):
		pass
		
	def wait(self):
		status = self.network.requests[0].wait(-1)
		return status
