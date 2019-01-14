import csv
import cv2
import numpy as np

lines = []
with open('training_data/data/driving_log8.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

augmented_images = []
augmented_measurements = []
for line in lines:
    for i in range(3):
        measurement = float(line[3])
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'training_data/data/IMG8/' + filename
        image = cv2.imread(current_path) # 0 = center, 1 = left, 2 = right
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented_images.append(image)
        augmented_images.append(cv2.flip(image, 1))
        
        correction = 0.2 # this is a parameter to tune
        if i == 1:  
            measurement = measurement + correction
        elif i == 2:
            measurement = measurement - correction
        augmented_measurements.append(measurement)
        augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print(X_train.shape)

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import optimizers

model = Sequential()
#Perform normalization and cropping of unnecessary features, output is now 80x320x3
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,20), (0,0))))

#First layer: Depth 24,output tensor is 40x160x24
model.add(Convolution2D(24, kernel_size = (5, 5), padding="same", activation = "relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

#2nd layer: Depth 36, output tensor is 20 x 80 x 36
model.add(Convolution2D(36, kernel_size = (5, 5), padding="same", activation = "relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

#3rd layer: Depth 48, output tensor is 10 x 40 x 48
model.add(Convolution2D(48, kernel_size = (5, 5), padding="same", activation = "relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

#4th layer: Depth 64, filter size 3x3 , output tensor is 5 x 20 x 64
model.add(Convolution2D(64, kernel_size = (3, 3), padding="same", activation = "relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

#4th layer: Depth 64, filter size 3x3 , output tensor is 3 x 18 x 64
model.add(Convolution2D(64, kernel_size = (3, 3), padding="valid", activation = "relu"))
model.add(Dropout(0.5))

#Flatten: 3456 neurons
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model = load_model('model2.h5')
adam = optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(loss='mse', optimizer=adam)
model.fit(X_train, y_train, batch_size = 126, validation_split=0.2, shuffle = True, epochs=10)

model.save('model2.h5')