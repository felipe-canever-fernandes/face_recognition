from argparse import ArgumentParser

from cv2 import dnn, dnn_Net


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

arguments = argument_parser.parse_args()

print("Loading face detector...")

detector: dnn_Net = dnn.readNetFromCaffe(
	arguments.prototxt,
	arguments.caffe_model,
)
