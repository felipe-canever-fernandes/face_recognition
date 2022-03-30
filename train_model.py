from argparse import ArgumentParser
from pickle import loads

argument_parser: ArgumentParser = ArgumentParser()

argument_parser.add_argument(
	"-e",
	"--embeddings",
	required=True,
	help="path to serialized database of facial embeddings",
)

arguments = argument_parser.parse_args()

print("Loading face embeddings...")

data: "dict[str, list]" = {}

with open(arguments.embeddings, "rb") as file:
	data = loads(file.read())
