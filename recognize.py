from argparse import ArgumentParser
from pickle import loads

from cv2 import dnn, dnn_Net, imread
import imutils
from numpy import ndarray
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


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
	"-le",
	"--label-encoder",
	required=True,
	help="path to label encoder",
)

argument_parser.add_argument(
	"-r",
	"--recognizer",
	required=True,
	help="path to model trained to recognize faces",
)

argument_parser.add_argument(
	"-i",
	"--image",
	required=True,
	help="path to input image",
)

arguments = argument_parser.parse_args()

print("Loading face detector...")

detector: dnn_Net = dnn.readNetFromCaffe(
	arguments.prototxt,
	arguments.caffe_model,
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
