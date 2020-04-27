"""
    Train the model

    Authors:
        Andrew Sanders
        Xavier Hodges
"""
import pandas as pd
import numpy as np
from keras import callbacks
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import io
import os

image_size = 100
target_size = (image_size, image_size)
image_shape = (image_size, image_size, 1)

datacsv = pd.read_csv('airbnb-listings.csv', sep=';', low_memory=False)
data = datacsv.query('`Number of Reviews` > 20').query('`Review Scores Value` > 9.0').query(
    '`Room Type` == "Entire home/apt"')
indexes = data.index.to_list()
final = data['Price'][indexes]

imlist = []

for file in os.listdir('D:/Pictures2/Train'):
    temp = ['D:/Pictures2/Train/' + str(file), final[int(file[:-4])]]
    imlist.append(temp)

TrainDataframe = pd.DataFrame(imlist, columns=['Image Data', 'Price'])
TrainDataframe = TrainDataframe.dropna()
imlist = []

for file in os.listdir('D:/Pictures2/Test'):
    temp = ['D:/Pictures2/Test/' + str(file), final[int(file[:-4])]]
    imlist.append(temp)

TestDataframe = pd.DataFrame(imlist, columns=['Image Data', 'Price'])
TestDataframe = TestDataframe.dropna()

scaler = MinMaxScaler()
TrainDataframe['Price'] = scaler.fit_transform(TrainDataframe[['Price']])
TestDataframe['Price'] = scaler.transform(TestDataframe[['Price']])


datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_dataframe(dataframe=TrainDataframe, x_col='Image Data', y_col='Price',
                                              class_mode='raw', target_size=target_size, batch_size=8,
                                              color_mode='grayscale')
test_generator = datagen.flow_from_dataframe(dataframe=TestDataframe, x_col='Image Data', y_col='Price',
                                             class_mode='raw', target_size=target_size, batch_size=8,
                                             color_mode='grayscale')

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size



# TWEAK ALL OF THIS
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=image_shape, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='adam',
              loss='mean_squared_error')


model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=test_generator,
                    validation_steps=STEP_SIZE_TEST, epochs=25, callbacks=callbacks.EarlyStopping(monitor='mean_squared_error', patience=2, restore_best_weights=True))

model.save('newmodel.h5')
model.summary()

# Use to test
# scaler.inverse_transform(model.predict(np.expand_dims(image.img_to_array(image.load_img('D:/Pictures2/Test/277.jpg', target_size=target_size, color_mode='grayscale')), axis=0)))[0][0]
