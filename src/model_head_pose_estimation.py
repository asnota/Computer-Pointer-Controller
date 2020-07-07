'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IECore, IENetwork
import numpy as np
import cv2
import logging

class Model_HeadPoseEstimation:
	'''
	Class for the Head Pose Estimation Model.
	'''
	def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
		self.model_structure = model_path
		self.model_weights = model_path.replace('.xml', '.bin')
		self.device_name = device
		self.threshold = threshold
		self.logger = logging.getLogger('hp')
		self.model_name = 'Head Pose Estimation Model'
		
		try:
			self.core = IECore()
			self.model = IENetwork(self.model_structure, self.model_weights)
		except Exception as e:
			self.logger.error("Error while initilizing" + str(self.model_name) + str(e))
			raise ValueError("Could not initialise the network. Have you enterred the correct model path?")
		
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
		try:
			preprocessed_image = self.preprocess_input(image)
			self.network.start_async(request_id, inputs={self.input_name: preprocessed_image})
			if self.wait() == 0:
				outputs = self.network.requests[0].outputs
				print("Outputs from head position", outputs)
				
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
		#print("Head position outputs: ", outputs[0])
		try:		
			pose_output.append(outputs['angle_y_fc'][0][0])				
			pose_output.append(outputs['angle_p_fc'][0][0])
			pose_output.append(outputs['angle_r_fc'][0][0])
		except Exception as e:
			self.logger.error("Error occured in preprocess_output() method of " + str(self.model_name) + str(e))
		return pose_output
	
	def wait(self):
		status = self.network.requests[0].wait(-1)
		return status
