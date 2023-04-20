import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAvgPool2D
from keras.models import Model
from keras.optimizers import Adam
import scipy

imgWidth, imgHeight = 299, 299

trainDataLocation = r"C:\Users\Eweaa\Desktop\Python\CNN3\Dataset\TRAIN"
validationDataLocation = r"C:\Users\Eweaa\Desktop\Python\CNN3\Dataset\VAL"


noTrainSamples = 65
noValidationSamples = 10
epochs = 20
batchSize = 5


trainDataGen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

valDataGen = ImageDataGenerator(rescale=1. / 255)

trainGenerator = trainDataGen.flow_from_directory(
    trainDataLocation,
    target_size=(imgHeight, imgWidth),
    batch_size=batchSize,
    class_mode='binary'
)

validationGenerator = trainDataGen.flow_from_directory(
    validationDataLocation,
    target_size=(imgHeight, imgWidth),
    batch_size=batchSize,
    class_mode='binary'
)

baseModel = applications.InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(imgWidth, imgHeight, 3)
)

modelTop = Sequential()
modelTop.add(GlobalAvgPool2D(input_shape=baseModel.output_shape[1:], data_format=None)),
modelTop.add(Dense(256, activation='relu'))
modelTop.add(Dropout(0.5))
modelTop.add(Dense(1, activation='sigmoid'))

model = Model(inputs=baseModel.input, outputs=modelTop(baseModel.output))

model.compile(optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    trainGenerator,
    steps_per_epoch=noTrainSamples // batchSize, epochs=epochs, validation_data=validationGenerator, validation_steps=noValidationSamples // batchSize)


import matplotlib.pyplot as plt

print(history.history.keys())

plt.figure()
plt.plot(history.history['accuracy'], 'orange', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'blue', label='Validation Accuracy')
plt.plot(history.history['loss'], 'red', label='Training Loss')
plt.plot(history.history['val_loss'], 'green', label='Validation Loss')
plt.legend()
plt.show()

import numpy as np
from keras.preprocessing import image
from keras.utils import load_img
#
# imgPath = r"C:\Users\Eweaa\Desktop\Python\CNN3\Dataset\TEST\Label 0\img1429.jpg"
# imgPath2 = r"C:\Users\Eweaa\Desktop\Python\CNN3\Dataset\TEST\Label 1\img1364.jpg"
# imgPath3 = r"C:\Users\Eweaa\Desktop\Python\CNN3\Dataset\TEST\Label 2\img1068.jpg"

# img = image.load_img(imgPath, target_size=(imgWidth, imgHeight))
# img = keras.utils.load_img(imgPath, target_size=(imgWidth, imgHeight))
# img2 = keras.utils.load_img(imgPath2, target_size=(imgWidth, imgHeight))
# img3 = keras.utils.load_img(imgPath3, target_size=(imgWidth, imgHeight))
# # img2 = image.load_img(imgPath2, target_size=(imgWidth, imgHeight))
#
# plt.imshow(img)
# plt.show()
#
# img = keras.utils.img_to_array(img)
# x = np.expand_dims(img, axis=0) * 1. / 255
# score1 = model.predict(x)
# print('first degree burn ', score1)

# print('Predicted: ', score, 'Chest X-ray' if score < 0.5 else 'Abd x-ray')

# plt.imshow(img2)
# plt.show()
#
# img2 = keras.utils.img_to_array(img2)
# x = np.expand_dims(img2, axis=0) * 1. / 255
# score2 = model.predict(x)
# print('second degree burn ', score2)
# print('Predicted: ', score2, 'Chest X-ray' if score2 < 0.5 else 'Abd x-ray')

# img3 = keras.utils.img_to_array(img3)
# x = np.expand_dims(img2, axis=0) * 1. / 255
# score3 = model.predict(x)
# print('third degree burn ', score3)
#
# plt.imshow(img3)
# plt.show()

# print('Predicted: ', score2, 'Chest X-ray' if score2 < 0.5 else 'Abd x-ray')


# import shutil
#
# for i in range(1440):
#     labelImgPath =

import shutil
from statistics import mean
from os.path import exists
arr = []
for x in range(1440):
    if exists(f'./Dataset/TRAIN/Label 0/img{x}.jpg'):
        img = keras.utils.load_img(r'./Dataset/TRAIN/Label 0/img'+str(x)+'.jpg', target_size=(imgWidth, imgHeight))
        img = keras.utils.img_to_array(img)
        x = np.expand_dims(img, axis=0) * 1. / 255
        score = model.predict(x)
        arr.append(score)
    else:
        print('does not exist')
print(arr)
theMean = mean(arr)
print('the threshold is ' + str(theMean))
