from argparse import ArgumentParser
from pickle import loads

from cv2 import dnn, dnn_Net
from sklearn.preprocessing import LabelEncoder


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

arguments = argument_parser.parse_args()

print("Loading face detector...")

detector: dnn_Net = dnn.readNetFromCaffe(
	arguments.prototxt,
	arguments.caffe_model,
)

print("Loading embedding model...")

embedder: dnn_Net = dnn.readNetFromTorch(arguments.embedding_model)

label_encoder: LabelEncoder = None

with open(arguments.label_encoder, "rb") as file:
	label_encoder = loads(file.read())
