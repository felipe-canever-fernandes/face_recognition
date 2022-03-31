from argparse import Namespace
from pickle import loads

import cv2
from cv2 import dnn, dnn_Net, imread
import imutils
from numpy import array, float32, ndarray
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

	Argument(
		long_flag="--confidence",
		short_flag="-c",
		help="minimum probability, to filter weak detections",
		type=float,
		default_value=0.5,
	),
)

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

image_height, image_width = image.shape[: 2]

print("Loading embedding model...")

embedder: dnn_Net = dnn.readNetFromTorch(arguments.embedding_model)

for i in range(detections.shape[2]):
	confidence: float32 = detections[0, 0, i, 2]

	if confidence < arguments.confidence:
		continue

	box: ndarray = detections[0, 0, i, 3 : 7]
	box *= array([image_width, image_height, image_width, image_height])
	start_x, start_y, end_x, end_y = box.astype("int")

	face: ndarray = image[start_y : end_y, start_x : end_x]
	face_height, face_width = face.shape[: 2]

	if face_width < 20 or face_height < 20:
		continue

	face_blob: ndarray = dnn.blobFromImage(
		image=face,
		scalefactor=1.0 / 255,
		size=(96, 96),
		mean=(0, 0, 0),
		swapRB=True,
		crop=False,
	)

	embedder.setInput(face_blob)
	embedding: ndarray = embedder.forward()
