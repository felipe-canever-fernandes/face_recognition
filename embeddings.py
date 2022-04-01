import cv2
from cv2 import dnn, dnn_Net, imread
import imutils
from numpy import array, ndarray

_MAXIMUM_IMAGE_WIDTH: int = 600
_IMAGE_BLOB_SIZE: "tuple[int, int]" = (300, 300)

_detector: dnn_Net = None

def initialize(prototxt_path: str, caffe_model_path: str):
	global _detector
	_detector = dnn.readNetFromCaffe(prototxt_path, caffe_model_path)

def process_image(image_path: str) -> "tuple[ndarray, ndarray]":
	image: ndarray = imread(image_path)
	image = imutils.resize(image, width=_MAXIMUM_IMAGE_WIDTH)

	image_blob: ndarray = dnn.blobFromImage(
		image=cv2.resize(image, _IMAGE_BLOB_SIZE),
		scalefactor=1.0,
		size=_IMAGE_BLOB_SIZE,
		mean=(104.0, 177.0, 123.0),
		swapRB=False,
		crop=False,
	)

	return image, image_blob

def detect_faces(image_blob: ndarray) -> ndarray:
	global _detector
	_detector.setInput(image_blob)
	return _detector.forward()

def get_face(
	image: ndarray,
	detections: ndarray,
	i_detection: int,
) -> "tuple[ndarray, tuple[int, int, int, int]]":
	image_height, image_width = image.shape[: 2]

	box: ndarray = detections[0, 0, i_detection, 3 : 7]
	box *= array([image_width, image_height, image_width, image_height])
	start_x, start_y, end_x, end_y = box.astype("int")

	face: ndarray = image[start_y : end_y, start_x : end_x]

	return face, (start_x, start_y, end_x, end_y)
