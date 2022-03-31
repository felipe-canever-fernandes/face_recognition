from argparse import Namespace
from pickle import dumps
from os import path

import cv2
from cv2 import dnn, dnn_Net, imread
import imutils
from imutils import paths
from numpy import argmax, array, float32, ndarray

from argument_parsing import Argument, get_arguments


arguments: Namespace = get_arguments(
	Argument(
		long_flag="--input",
		short_flag="-i",
		help="path to input directory of face images",
		is_required=True,
	),

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
		long_flag="--confidence",
		short_flag="-c",
		help="minimum probability, to filter weak detections",
		type=float,
		default_value=0.5,
	),

	Argument(
		long_flag="--embeddings",
		short_flag="-e",
		help="path to output serialized database of facial embeddings",
		is_required=True,
	),
)

image_paths: "list[str]" = list(paths.list_images(arguments.input))
MODEL_IMAGE_SIZE: "tuple[int, int]" = (300, 300)

print("Loading face detector...")

detector: dnn_Net = dnn.readNetFromCaffe(
	arguments.prototxt,
	arguments.caffe_model,
)

print("Loading embedding model...")

embedder: dnn_Net = dnn.readNetFromTorch(arguments.embedding_model)

print("Quantifying faces...")

embeddings: "list[ndarray]" = []
names: "list[str]" = []

for i_image, image_path in enumerate(image_paths):
	print(f"Processing image {i_image + 1}/{len(image_paths)}...")

	image: ndarray = imread(image_path)
	image = imutils.resize(image, width=600)

	image_blob: ndarray = dnn.blobFromImage(
		image=cv2.resize(image, MODEL_IMAGE_SIZE),
		scalefactor=1.0,
		size=MODEL_IMAGE_SIZE,
		mean=(104.0, 177.0, 123.0),
		swapRB=False,
		crop=False,
	)

	detector.setInput(image_blob)
	detections: ndarray = detector.forward()
	
	if len(detections) <= 0:
		continue

	i_maximum_confidence: int = argmax(detections[0, 0, :, 2])
	confidence: float32 = detections[0, 0, i_maximum_confidence, 2]

	if confidence < arguments.confidence:
		continue

	image_height, image_width = image.shape[: 2]

	box: ndarray = detections[0, 0, i_maximum_confidence, 3 : 7]
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
	embeddings.append(embedding.flatten())

	face_name: str = image_path.split(path.sep)[-2]
	names.append(face_name)

print(f"Serializing {len(names)} embeddings...")

data: "dict[str, list]" = {
	"names": names,
	"embeddings": embeddings,
}

with open(arguments.embeddings, "wb") as file:
	file.write(dumps(data))
