import pickle
from io import BytesIO
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import sys
import re
import base64
import json
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

# TensorFlow and tf.keras
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


app = Flask(__name__)

with open("amenities.json") as f:
    amenities = json.load(f)

model = load_model('newmodel.h5')
with open('linearRegression_model.pkl', 'rb') as file:
    regr = pickle.load(file)
with open('linearRegressionScaler.pkl', 'rb') as file:
    regression_scaler = pickle.load(file)


def model_predict(img, model):
    image_size = 100
    target_size = (image_size, image_size)

    preds = model.predict(np.expand_dims(image.img_to_array(image.load_img(
        img, target_size=target_size, color_mode='grayscale')), axis=0))

    return str(preds[0][0])


@app.route("/", methods=['GET'])
def index():
    return render_template('index.html', amenities=amenities)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = request.json

        image_data = re.sub('^data:image/.+;base64,', '', img)
        pil_image = BytesIO(base64.b64decode(image_data))

        prediction = model_predict(pil_image, model)

        return jsonify(prediction=prediction)

    return None


@app.route("/regress", methods=['GET', 'POST'])
def regress():
    if request.method == "POST":
        params = request.json

        regressRes = regression_scaler.inverse_transform(
            regr.predict([params]))[0][0]

        return jsonify(regressRes=regressRes)


if __name__ == "__main__":
    app.run(debug=True)
