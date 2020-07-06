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
			self.logger.error("Error While Initilizing" + str(self.model_name) + str(e))
			raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
		
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

	def predict(self, image, request_id=0):
		try:
			prepocessed_image = self.preprocess_input(image)
			self.network.start_async(request_id, inputs={self.input_name: preprocessed_image})
			if self.wait() == 0:
				outputs = self.network.requests[0].outputs[self.output_name]
				coords, cropped_image = self.preprocess_output(outputs, image)
		except Exception as e:
			self.logger.error("Error occured in predict() method of the Model_FaceDetection class")
		return coords, cropped_image


	def preprocess_input(self, image):
		try:
			image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
			image = image.transpose((2, 0, 1))
			image = image.reshape(1, *image.shape)
		except Exception as e:
			self.logger.error("Error While preprocessing Image in " + str(self.model_name) + str(e))
		return image


	def preprocess_output(self, outputs):
		width, height = int(image.shape[1]), int(image.shape[0])
		detections = []
		cropped_image = image
		coords = np.squeeze(coords)
		try:
			for coord in coords:
				image_id, label, threshold, xmin, ymin, xmax, ymax = coord
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
			self.logger.error("Error While drawing bounding boxes on image in Face Detection Model" + str(e))
		return detections, cropped_image
		
	def wait(self):
		status = self.exec_network.requests[0].wait(-1)
		return status
