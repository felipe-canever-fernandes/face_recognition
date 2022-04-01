from argparse import Namespace
from pickle import loads

from cv2 import FONT_HERSHEY_SIMPLEX, dnn, dnn_Net, imshow, putText
from cv2 import rectangle, waitKey
from numpy import argmax, array, float32, float64, ndarray
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from argument_parsing import get_arguments
from arguments import CAFFE_MODEL, CONFIDENCE, EMBEDDING_MODEL, IMAGE
from arguments import LABEL_ENCODER, PROTOTXT, RECOGNIZER
from embeddings import detect_faces, initialize, process_image


arguments: Namespace = get_arguments(
	PROTOTXT,
	CAFFE_MODEL,
	IMAGE,
	EMBEDDING_MODEL,
	RECOGNIZER,
	LABEL_ENCODER,
	CONFIDENCE,
)

print("Loading face detector...")
initialize(arguments.prototxt, arguments.caffe_model)

image, image_blob = process_image(arguments.image)
detections: ndarray = detect_faces(image_blob)

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

for i_detection in range(detections.shape[2]):
	confidence: float32 = detections[0, 0, i_detection, 2]

	if confidence < arguments.confidence:
		continue

	box: ndarray = detections[0, 0, i_detection, 3 : 7]
	box *= array([image_width, image_height, image_width, image_height])
	start_x, start_y, end_x, end_y = box.astype("int")

	face: ndarray = image[start_y : end_y, start_x : end_x]
	face_height, face_width = face.shape[: 2]

	if face_width < 20 or face_height < 20:
		continue

	rectangle(
		image,
		(start_x, start_y),
		(end_x, end_y),
		color=RED,
		thickness=2,
	)

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

	i_maximum_probability: int = argmax(predictions)

	name: str = label_encoder.classes_[i_maximum_probability]
	probability: float64 = predictions[i_maximum_probability]

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
