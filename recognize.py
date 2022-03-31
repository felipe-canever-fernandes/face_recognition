from argparse import Namespace
from pickle import loads

import cv2
from cv2 import dnn, dnn_Net, imread
import imutils
from numpy import ndarray
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from argument_parsing import Argument, get_arguments


arguments: Namespace = get_arguments(
	Argument(
		long_flag="--prototxt",
		short_flag="-p",
		help="path to Caffe prototxt file",
		is_required=True,
	),

	Argument(
		long_flag="--caffe-model",
		short_flag="-cm",
		help="path to Caffe pre-trained model",
		is_required=True,
	),

	Argument(
		long_flag="--embedding-model",
		short_flag="-em",
		help="path to OpenCV's deep learning face embedding model",
		is_required=True,
	),

	Argument(
		long_flag="--label-encoder",
		short_flag="-le",
		help="path to label encoder",
		is_required=True,
	),

	Argument(
		long_flag="--recognizer",
		short_flag="-r",
		help="path to model trained to recognize faces",
		is_required=True,
	),

	Argument(
		long_flag="--image",
		short_flag="-i",
		help="path to input image",
		is_required=True,
	),
)

print("Loading embedding model...")

embedder: dnn_Net = dnn.readNetFromTorch(arguments.embedding_model)

def read_data(path: str):
	with open(path, "rb") as file:
		return loads(file.read())

print("Loading face recognizer...")

label_encoder: LabelEncoder = read_data(arguments.label_encoder)
recognizer: SVC = read_data(arguments.recognizer)

image: ndarray = imread(arguments.image)
image = imutils.resize(image, width=600)

MODEL_IMAGE_SIZE: "tuple[int, int]" = (300, 300)

image_blob: ndarray = dnn.blobFromImage(
	image=cv2.resize(image, MODEL_IMAGE_SIZE),
	scalefactor=1.0,
	size=MODEL_IMAGE_SIZE,
	mean=(104.0, 177.0, 123.0),
	swapRB=False,
	crop=False,
)

print("Loading face detector...")

detector: dnn_Net = dnn.readNetFromCaffe(
	arguments.prototxt,
	arguments.caffe_model,
)

detector.setInput(image_blob)
detections: ndarray = detector.forward()
