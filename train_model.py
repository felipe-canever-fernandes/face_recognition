from argparse import Namespace
from pickle import dumps, loads

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from numpy import ndarray

from argument_parsing import get_arguments
from arguments import EMBEDDINGS, LABEL_ENCODER, RECOGNIZER


arguments: Namespace = get_arguments(
	EMBEDDINGS,
	LABEL_ENCODER,
	RECOGNIZER,
)

print("Loading face embeddings...")

data: "dict[str, list]" = {}

with open(arguments.embeddings, "rb") as file:
	data = loads(file.read())

print("Encoding labels...")

label_encoder: LabelEncoder = LabelEncoder()
labels: ndarray = label_encoder.fit_transform(data["names"])


def write_data(path: str, object) -> None:
	with open(path, "wb") as file:
		file.write(dumps(object))


write_data(arguments.label_encoder, label_encoder)

print("Training model...")

recognizer: SVC = SVC(
	C=1.0,
	kernel="linear",
	probability=True,
)

recognizer.fit(data["embeddings"], labels)

write_data(arguments.recognizer, recognizer)
