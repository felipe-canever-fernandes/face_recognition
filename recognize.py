from argparse import Namespace
from pickle import loads

import cv2
from cv2 import FONT_HERSHEY_SIMPLEX, dnn, dnn_Net, imread, imshow, putText
from cv2 import rectangle, waitKey
import imutils
from numpy import argmax, array, float32, float64, imag, ndarray
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


def read_data(path: str):
	with open(path, "rb") as file:
		return loads(file.read())


print("Loading face recognizer...")

recognizer: SVC = read_data(arguments.recognizer)
label_encoder: LabelEncoder = read_data(arguments.label_encoder)

RED: "tuple[int, int, int]" = (0, 0, 255)

for detection_index in range(detections.shape[2]):
	confidence: float32 = detections[0, 0, detection_index, 2]

	if confidence < arguments.confidence:
		continue

	box: ndarray = detections[0, 0, detection_index, 3 : 7]
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

	predictions: ndarray = recognizer.predict_proba(embedding)[0]

	maximum_probability_index: int = argmax(predictions)

	rectangle(
		image,
		(start_x, start_y),
		(end_x, end_y),
		color=RED,
		thickness=2,
	)

	name: str = label_encoder.classes_[maximum_probability_index]
	probability: float64 = predictions[maximum_probability_index]

	y = start_y - 10 if start_y - 10 > 10 else start_y + 10

	putText(
		image,
		f"{name}: {probability * 100:.2f}%",
		(start_x, y),
		FONT_HERSHEY_SIMPLEX,
		fontScale=0.45,
		color=RED,
		thickness=2,
	)

imshow("Image", image)
waitKey(0)
