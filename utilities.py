from pickle import dumps, loads


def write_data(path: str, object) -> None:
	with open(path, "wb") as file:
		file.write(dumps(object))
		

def read_data(path: str):
	with open(path, "rb") as file:
		return loads(file.read())
