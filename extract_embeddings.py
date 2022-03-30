from argparse import ArgumentParser
from os import path

from cv2 import dnn, dnn_Net, imread
import imutils
from imutils import paths
from numpy import ndarray

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
