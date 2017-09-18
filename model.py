import cv2
import csv
import numpy as np
from PIL import Image
import random

def show(filename):
    img = Image.open('../data/IMG/' + filename)
    img.save('original.jpg')

    right = Image.open('../data/IMG/' + filename.replace('center', 'right'))
    right.save('right.jpg')

    left = Image.open('../data/IMG/' + filename.replace('center', 'left'))
    left.save('left.jpg')

    width, height = img.size

    img1 = img.crop((70, 25, width, height))
    img1.save('cropped.jpg')

    img2 = img1.convert('L')
    img2.save('normalized.jpg')

    img3 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    img3.save('flipped.jpg')


lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
measurements = []

randidx = random.randint(0, len(lines))
source_path = lines[randidx][0]
file_name = source_path.split('/')[-1]
show(file_name)
print('Saved files')


change = [0.0, 0.2, -0.2]
for idx, line in enumerate(lines):
    for i in range(3):
        source_path = line[i]
        file_name = source_path.split('/')[-1]
        current_path = '../data/IMG/'+file_name
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement + change[i])

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Building a basic NN
# Has a single output that predicts the steering angle
from keras.layers.core import K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers import Cropping2D
from keras.layers.core import Dropout

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

model.add(Cropping2D(cropping=((70,25), (0,0)))) # adding cropping
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.7))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.summary()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20, verbose=1)
model.save('model.h5')
