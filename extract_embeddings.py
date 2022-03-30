from argparse import ArgumentParser
from pickle import dumps
from os import path

import cv2
from cv2 import dnn, dnn_Net, imread
import imutils
from imutils import paths
from numpy import argmax, array, float32, ndarray


argument_parser: ArgumentParser = ArgumentParser()

argument_parser.add_argument(
	"-p",
	"--prototxt",
	required=True,
	help="path to Caffe prototxt file",
)

argument_parser.add_argument(
	"-cm",
	"--caffe-model",
	required=True,
	help="path to Caffe pre-trained model",
)

argument_parser.add_argument(
	"-em",
	"--embedding-model",
	required=True,
	help="path to OpenCV's deep learning face embedding model",
)

argument_parser.add_argument(
	"-i",
	"--input",
	required=True,
	help="path to input directory of face images",
)

argument_parser.add_argument(
	"-c",
	"--confidence",
	type=float,
	default=0.5,
	help="minimum probability, to filter weak detections",
)

argument_parser.add_argument(
	"-e",
	"--embeddings",
	required=True,
	help="path to output serialized database of facial embeddings",
)

arguments = argument_parser.parse_args()

print("Loading face detector...")

detector: dnn_Net = dnn.readNetFromCaffe(
	arguments.prototxt,
	arguments.caffe_model,
)

print("Loading embedding model...")

embedder: dnn_Net = dnn.readNetFromTorch(arguments.embedding_model)

print("Quantifying faces...")

image_paths: "list[str]" = list(paths.list_images(arguments.input))

MODEL_IMAGE_SIZE: "tuple[int, int]" = (300, 300)

embeddings: "list[ndarray]" = []
names: "list[str]" = []

for image_index, image_path in enumerate(image_paths):
	print(f"Processing image {image_index + 1}/{len(image_paths)}...")

	face_name: str = image_path.split(path.sep)[-2]

	image: ndarray = imread(image_path)
	image = imutils.resize(image, width=600)

	image_height, image_width = image.shape[: 2]

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

	maximum_confidence_index: int = argmax(detections[0, 0, :, 2])
	confidence: float32 = detections[0, 0, maximum_confidence_index, 2]

	if confidence < arguments.confidence:
		continue

	box: ndarray = detections[0, 0, maximum_confidence_index, 3 : 7]
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

	names.append(face_name)
	embeddings.append(embedding.flatten())

print(f"Serializing {len(names)} embeddings...")

data: "dict[str, list]" = {
	"names": names,
	"embeddings": embeddings,
}

with open(arguments.embeddings, "wb") as file:
	file.write(dumps(data))
