from argparse import ArgumentParser, Namespace
from typing import Type


class Argument:
	def __init__(
		self,
		long_flag: str,
		short_flag: str,
		help: str,
		is_required: bool = False,
		type: Type = str,
		default_value = None,
	):
		self.short_flag = short_flag
		self.long_flag = long_flag
		self.is_required = is_required
		self.type = type
		self.default_value = default_value
		self.help = help


def get_arguments(*arguments: Argument) -> Namespace:
	argument_parser: ArgumentParser = ArgumentParser()

	for argument in arguments:
		argument_parser.add_argument(
			argument.short_flag,
			argument.long_flag,
			required=argument.is_required,
			type=argument.type,
			default=argument.default_value,
			help=argument.help,
		)
	
	return argument_parser.parse_args()
