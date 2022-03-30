from argparse import ArgumentParser
from pickle import dumps, loads

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from numpy import ndarray

argument_parser: ArgumentParser = ArgumentParser()

argument_parser.add_argument(
	"-e",
	"--embeddings",
	required=True,
	help="path to serialized database of facial embeddings",
)

argument_parser.add_argument(
	"-r",
	"--recognizer",
	required=True,
	help="path to output model trained to recognize faces",
)

argument_parser.add_argument(
	"-le",
	"--label-encoder",
	required=True,
	help="path to output label encoder",
)

arguments = argument_parser.parse_args()

print("Loading face embeddings...")

data: "dict[str, list]" = {}

with open(arguments.embeddings, "rb") as file:
	data = loads(file.read())

print("Encoding labels...")

label_encoder: LabelEncoder = LabelEncoder()

def write_data(path: str, object) -> None:
	with open(path, "wb") as file:
		file.write(dumps(object))

write_data(arguments.label_encoder, label_encoder)

labels: ndarray = label_encoder.fit_transform(data["names"])

print("Training model...")

recognizer: SVC = SVC(
	C=1.0,
	kernel="linear",
	probability=True,
)

recognizer.fit(data["embeddings"], labels)

write_data(arguments.recognizer, recognizer)
