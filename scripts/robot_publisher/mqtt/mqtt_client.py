
from urllib.parse import urlparse
import paho.mqtt.client as mqtt
import time
import yaml
from yaml.parser import ParserError, ScannerError
import sys
import json
import threading
import time

FIRST_RECONNECT_DELAY = 1
RECONNECT_RATE = 2
MAX_RECONNECT_COUNT = 12
MAX_RECONNECT_DELAY = 60

class MQTT:

	def __init__(self):
		self.asr_topic = None
		self.vad_topic = None
		self.data = self.load_config()
	
	def load_config(self):

		try:
			with open("mqtt_config_python.yaml", "r") as yamlfile:
				data = yaml.load(yamlfile, Loader=yaml.FullLoader)
				self.human_detection_topic = "/topic/testing/testroom/realsense_human_detection"
				print("Publishing to Human detection", self.human_detection_topic)
				return data
		except FileNotFoundError as e:
			print("Config file does not exist!")
		except ParserError as e:
			print("YAML file is not correct!")
			print(e)
		except ScannerError as e:
			print("YAML file is not correct!")
			print(e)
		return None

	def connect(self):

		username = self.data["MQTTUSERNAME"]
		password = self.data["MQTTPASSWORD"]
		host = self.data["MQTTADDRESS"]
		path = self.data["MQTTPATH"]
		client_id = "operator_pc"
		port = 443 # for websockets
		self.topics = self.data["TOPICS"]
		self.connected = False
		self.client = mqtt.Client(client_id=client_id, clean_session=True, transport="websockets")
		self.client.tls_set()
		self.client.username_pw_set(username, password)
		self.client.on_connect = self.on_connect
		self.client.on_disconnect = self.on_disconnect
		self.client.ws_set_options(path=path)
		self.client.connect(host, port)
		self.connected = True
		self.client.on_message = self.receive_message
		self.client.loop_forever()


	def on_connect(self, client, userdata, flags, rc):
		print("Connected to MQTT")
	
	def receive_message(self, client, userdata, msg):
		pass

	def on_disconnect(self, client, userdata, rc):

		print("Disconnected with result code: %s", rc)
		self.connected = False
		reconnect_count, reconnect_delay = 0, FIRST_RECONNECT_DELAY

		while reconnect_count < MAX_RECONNECT_COUNT:
			print("Reconnecting in %d seconds...", reconnect_delay)
			time.sleep(reconnect_delay)

			try:
				self.client.reconnect()
				print("Reconnected successfully!")
				self.connected = True
				self.subscribe()
				return
			except Exception as err:
				print("%s. Reconnect failed. Retrying...", err)

			reconnect_delay *= RECONNECT_RATE
			reconnect_delay = min(reconnect_delay, MAX_RECONNECT_DELAY)
			reconnect_count += 1
		print("Reconnect failed after %s attempts. Exiting...", reconnect_count)


	def publish_human_results(self, message):
		if self.human_detection_topic != None:
			# print(self.vad_topic, message)
			self.client.publish(self.human_detection_topic, message)
