from argparse import Namespace

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from numpy import ndarray

from argument_parsing import get_arguments
from arguments import EMBEDDINGS, LABEL_ENCODER, RECOGNIZER
from utilities import read_data, write_data


arguments: Namespace = get_arguments(
	EMBEDDINGS,
	LABEL_ENCODER,
	RECOGNIZER,
)

print("Loading face embeddings...")
data: "dict[str, list]" = read_data(arguments.embeddings)

print("Encoding labels...")

label_encoder: LabelEncoder = LabelEncoder()
labels: ndarray = label_encoder.fit_transform(data["names"])
write_data(arguments.label_encoder, label_encoder)

print("Training model...")

recognizer: SVC = SVC(
	C=1.0,
	kernel="linear",
	probability=True,
)

recognizer.fit(data["embeddings"], labels)

write_data(arguments.recognizer, recognizer)
