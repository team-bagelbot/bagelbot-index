from google.cloud import logging

class AppLogger():
	def __init__(self, name):
		self.name = name
		logging_client = logging.Client()
		self.client = logging_client.logger(name)

	def __str__(self):
		return f"name: {self.name}"