from flask import Flask
import json

app = Flask(__name__)

@app.route('/')
def index():
	return 'Hello from AWS!'

@app.route('/predict')
def predict():
	with open("temp_json_file.json","r") as jsonfile:
		json_object = json.load(jsonfile)
	json_return = json.dumps(json_object)
	return json_return
