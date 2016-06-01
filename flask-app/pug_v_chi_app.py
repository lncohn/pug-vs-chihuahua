import numpy as np
from skimage import io
import os
from flask import Flask
from flask import render_template
from flask import abort, jsonify, request
from keras.models import model_from_json, Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import requests

app = Flask(__name__)


def Lee(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3, 224, 224), trainable=False))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2), trainable=False))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2), trainable=False))

    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2), trainable=False))

    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2), trainable=False))

    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2), trainable=False))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', trainable=False))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', trainable=False))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

model = Lee('chi_model_weights.h5')

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/', methods=['POST'])
def score():
    url = request.form['text']
    f = open('my_image2.jpg','wb')
    f.write(requests.get(url).content)
    f.close()
    os.system("convert my_image2.jpg -resize 224x224^ -gravity Center -crop 224x224+0+0 +repage my_image2_cropped.jpg")
    my_img = io.imread("my_image2_cropped.jpg").transpose()
    img = my_img.transpose()
    img = np.array(img).transpose()
    img = img.reshape((1, 3, 224, 224))
    pug_score = 100*model.predict(img)
    chi_score = 100 - pug_score
    return render_template('result.html', image_url=url, pug_score=str(pug_score[0, 1]), chi_score=str(chi_score[0, 1]))

if __name__ == '__main__':
	app.run(port=5000,debug=True)



