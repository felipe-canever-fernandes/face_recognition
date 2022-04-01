from pickle import dumps

def write_data(path: str, object) -> None:
	with open(path, "wb") as file:
		file.write(dumps(object))
