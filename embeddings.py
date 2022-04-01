from numpy import ndarray

import cv2
from cv2 import dnn, dnn_Net, imread
import imutils

MAXIMUM_IMAGE_WIDTH: int = 600
IMAGE_BLOB_SIZE: "tuple[int, int]" = (300, 300)

detector: dnn_Net = None

def initialize(prototxt_path: str, caffe_model_path: str):
	global detector
	detector = dnn.readNetFromCaffe(prototxt_path, caffe_model_path)

def process_image(image_path: str) -> "tuple[ndarray, ndarray]":
	image: ndarray = imread(image_path)
	image = imutils.resize(image, width=MAXIMUM_IMAGE_WIDTH)

	image_blob: ndarray = dnn.blobFromImage(
		image=cv2.resize(image, IMAGE_BLOB_SIZE),
		scalefactor=1.0,
		size=IMAGE_BLOB_SIZE,
		mean=(104.0, 177.0, 123.0),
		swapRB=False,
		crop=False,
	)

	return image, image_blob

def detect_faces(image_blob: ndarray) -> ndarray:
	global detector
	detector.setInput(image_blob)
	return detector.forward()
