from argparse import Namespace
from os import path

from imutils import paths
from numpy import argmax, float32, ndarray

from argument_parsing import get_arguments
from arguments import CAFFE_MODEL, CONFIDENCE, DATASET, EMBEDDING_MODEL
from arguments import EMBEDDINGS, PASS_COUNT, PROTOTXT
from embeddings import detect_faces, extract_embedding, get_face, initialize
from embeddings import process_image
from utilities import write_data


arguments: Namespace = get_arguments(
	DATASET,
	PROTOTXT,
	CAFFE_MODEL,
	EMBEDDING_MODEL,
	CONFIDENCE,
	PASS_COUNT,
	EMBEDDINGS,
)

image_paths: "list[str]" = list(paths.list_images(arguments.dataset))

print("Loading face detector and embedding model...")
initialize(arguments.prototxt, arguments.caffe_model, arguments.embedding_model)

print("Quantifying faces...")

embeddings: "list[ndarray]" = []
names: "list[str]" = []

for i_image, image_path in enumerate(image_paths):
	print(f"Processing image {i_image + 1}/{len(image_paths)}...")

	image, image_blob = process_image(image_path)
	detections: ndarray = detect_faces(image_blob)
	
	if len(detections) <= 0:
		continue

	i_maximum_confidence: int = argmax(detections[0, 0, :, 2])
	confidence: float32 = detections[0, 0, i_maximum_confidence, 2]

	if confidence < arguments.confidence:
		continue

	face, _ = get_face(image, detections, i_maximum_confidence)
	face_height, face_width = face.shape[: 2]

	if face_width < 20 or face_height < 20:
		continue

	embedding: ndarray = extract_embedding(face)
	embeddings.append(embedding.flatten())

	face_name: str = image_path.split(path.sep)[-2]
	names.append(face_name)

names *= arguments.pass_count
embeddings *= arguments.pass_count

print(f"Serializing {len(names)} embeddings...")

data: "dict[str, list]" = {
	"names": names,
	"embeddings": embeddings,
}

write_data(arguments.embeddings, data)
