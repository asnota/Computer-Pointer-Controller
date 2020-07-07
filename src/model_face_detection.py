'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IECore, IENetwork
import numpy as np
import cv2
import logging

class Model_FaceDetection:
	'''
	Class for the Face Detection Model.
	'''
	def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
		self.model_structure = model_path
		self.model_weights = model_path.replace('.xml', '.bin')
		self.device_name = device
		self.threshold = threshold
		self.logger = logging.getLogger('fd')
		self.model_name = 'Face Detection Model'
		
		try:
			self.core = IECore()
			self.model = IENetwork(self.model_structure, self.model_weights)
		except Exception as e:
			self.logger.error("Error while iitilizing" + str(self.model_name) + str(e))
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
			self.logger.error("Error occured in load_model() method of "+str(self.model_name)+str(e))
			
	def predict(self, image, request_id=0):
		preprocessed_image = self.preprocess_input(image)	
		self.network.start_async(request_id, inputs={self.input_name: preprocessed_image})			
		
		try:
			if self.wait() == 0:
				outputs = self.network.requests[0].outputs[self.output_name]
				prediction, cropped_image = self.preprocess_output(outputs, image)				
		except Exception as e:
			self.logger.error("Error occured in predict() method of " + str(self.model_name) + str(e))
		return prediction, cropped_image


	def preprocess_input(self, image):
		try:
			image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
			image = image.transpose((2, 0, 1))
			image = image.reshape(1, *image.shape)
		except Exception as e:
			self.logger.error("Error occured in preprocess_input() method of " + str(self.model_name) + str(e))
		return image


	def preprocess_output(self, outputs, image):
		width, height = int(image.shape[1]), int(image.shape[0])			
		detections = []
		cropped_image = image		
		coords = np.squeeze(outputs)
		print("Coords shape: ", coords.shape)
		try:
			for coord in coords:				
				try:
					image_id, label, threshold, xmin, ymin, xmax, ymax = coord
				except Exception as e:
					self.logger.error("Error occured in preprocess_output() method1.1 of " + str(self.model_name) + str(e))	
				if image_id == -1:
					break
				if label == 1 and threshold >= self.threshold:
					xmin = int(xmin * width)
					ymin = int(ymin * height)
					xmax = int(xmax * width)
					ymax = int(ymax * height)
					detections.append([xmin, ymin, xmax, ymax])
					cropped_image = image[ymin:ymax, xmin:xmax]					
		except Exception as e:
			self.logger.error("Error occured in preprocess_output() method of " + str(self.model_name) + str(e))
		return detections, cropped_image
		
	def wait(self):
		status = self.network.requests[0].wait(-1)
		return status
