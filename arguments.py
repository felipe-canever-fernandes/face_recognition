from argument_parsing import Argument

CAFFE_MODEL: Argument = Argument(
	long_flag="--caffe-model",
	short_flag="-cm",
	help="path to the pre-trained Caffe model",
	is_required=True,
)

CONFIDENCE: Argument = Argument(
	long_flag="--confidence",
	short_flag="-c",
	help="the minimum probability, to filter weak detections",
	type=float,
	default_value=0.5,
)

DATASET: Argument = Argument(
	long_flag="--dataset",
	short_flag="-d",
	help="path to the directory containing each subject's folder",
	is_required=True,
)

EMBEDDING_MODEL: Argument = Argument(
	long_flag="--embedding-model",
	short_flag="-em",
	help="path to OpenCV's deep-learning face-embedding model",
	is_required=True,
)

EMBEDDINGS: Argument = Argument(
	long_flag="--embeddings",
	short_flag="-e",
	help="path to the output serialized database of facial embeddings",
	is_required=True,
)

IMAGE: Argument = Argument(
	long_flag="--image",
	short_flag="-i",
	help="path to the input image",
	is_required=True,
)

LABEL_ENCODER: Argument = Argument(
	long_flag="--label-encoder",
	short_flag="-le",
	help="path to the output label encoder",
	is_required=True,
)

PASS_COUNT: Argument = Argument(
	long_flag="--pass-count",
	short_flag="-pc",
	help="the number of times the model will be trained with the same embeddings",
	type=int,
	default_value=1,
)

PROTOTXT: Argument = Argument(
	long_flag="--prototxt",
	short_flag="-p",
	help="path to the Caffe prototxt file",
	is_required=True,
)

RECOGNIZER: Argument = Argument(
	long_flag="--recognizer",
	short_flag="-r",
	help="path to the output model trained to recognize faces",
	is_required=True,
)
