from io import BytesIO
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import sys
import re
import base64
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

# TensorFlow and tf.keras
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


app = Flask(__name__)


model = load_model('newmodel.h5')


def model_predict(img, model):
    image_size = 100
    target_size = (image_size, image_size)

    preds = model.predict(np.expand_dims(image.img_to_array(image.load_img(
        img, target_size=target_size, color_mode='grayscale')), axis=0))

    return str(preds[0][0])


@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')


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

        regressRes = params * 2

        return jsonify(regressRes=regressRes)


if __name__ == "__main__":
    app.run(debug=True)
