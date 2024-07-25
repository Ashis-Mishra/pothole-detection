# Pothole Detection and Severity Classification using CNN 
Introduction 
This project aims to detect potholes in road images and classify the severity of the potholes using a Convolutional Neural Network (CNN). The steps below outline the process from importing libraries and reading data to training the model.
Importing Libraries and Reading Data 
```python
import pandas as pd
import numpy as np

features = pd.read_csv(r"/content/Pothole-Detection/Dataset_Info.csv")
features.head(10)
```
Imports necessary libraries.
Reads a CSV file containing information about the dataset.
## Handling Missing Data 
```python
features.isna().sum()
features = features.drop('Unnamed: 4', axis=1)
features.head(10)
```
Checks for missing values and removes an unnecessary column. 
## Preparing Data 
```python
all_image_names = features['Image ID']
features = features.drop('Image ID', axis=1)
pothole_or_not = features['Pothole']
pothole_info = features['Number of Potholes']
pothole_level = features['Level']

pothole_or_not = pd.get_dummies(pothole_or_not, columns=['Pothole'])
pothole_or_not.columns = ['Normal Road', 'Pothole']
pothole_level = pd.get_dummies(pothole_level, columns=['Level'])

features = pothole_or_not.join(pothole_info)
features = features.join(pothole_level)

print(features.head(10))
```
Extracts and processes relevant columns.
Converts categorical columns to dummy/one-hot encoded columns. 
## Shuffling Data
```python
from sklearn.utils import shuffle

image_names_shuffled, labels_shuffled = shuffle(all_image_names, features)
```
Shuffles the data to ensure random distribution. 
## Splitting Data 
```python
from sklearn.model_selection import train_test_split

X_train_image_names, X_test_image_names, y_train, y_test = train_test_split(
    image_names_shuffled, labels_shuffled, test_size=0.3, random_state=1)
```
Splits the data into training and testing sets. 
## Converting Data to Numpy Arrays 
```python
train_pothole_or_not = np.array(train_pothole_or_not)
train_pothole_info = np.array(train_pothole_info)
train_pothole_level = np.array(train_pothole_level)

test_pothole_or_not = np.array(test_pothole_or_not)
test_pothole_info = np.array(test_pothole_info)
test_pothole_level = np.array(test_pothole_level)
```
Converts the training and testing sets to numpy arrays. 
## Image Preprocessing Function 
```python
import cv2

def get_image(file_location):
    img = cv2.imread(file_location)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = cv2.resize(img, (200, 200))
    return img
```
Defines a function to read, convert to grayscale, normalize, and resize images. 
## Custom Data Generator 
```python
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class My_Custom_Generator(Sequence):
    def __init__(self, image_filenames, pothole, pothole_number, pothole_level, batch_size=128):
        self.image_filenames = image_filenames
        self.pothole = pothole
        self.pothole_number = pothole_number
        self.pothole_level = pothole_level
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        a = self.pothole[idx * self.batch_size: (idx + 1) * self.batch_size]
        c = self.pothole_level[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = [np.array(a), np.array(c)]
        return np.array([
            get_image(f'/content/Pothole-Detection/UnifiedDataset/{str(file_name)}.jpg')
            for file_name in batch_x]), batch_y
```
Implements a custom data generator for loading images and corresponding labels in batches. 
## Creating Data Generators 
```python
batch_size = 60

my_training_batch_generator = My_Custom_Generator(X_train_image_names, train_pothole_or_not, train_pothole_info, train_pothole_level, batch_size)
my_validation_batch_generator = My_Custom_Generator(X_test_image_names, test_pothole_or_not, test_pothole_info, test_pothole_level, batch_size)
```
Initializes the custom data generators for training and validation. 
## Building the Model 
```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input, Activation, Add, BatchNormalization

input_ = Input(shape=(200, 200, 1), name="Input Layer")

conv_1 = Conv2D(32, kernel_size=(4, 4), name="conv_1")(input_)
act_1 = Activation("relu", name="act_1")(conv_1)
pool_1 = MaxPooling2D(pool_size=(8, 8), strides=(1, 1), padding="valid", name="pool_1")(act_1)

conv_2 = Conv2D(64, kernel_size=(8, 8), name="conv_2")(pool_1)
act_2 = Activation("relu", name="act_2")(conv_2)
pool_2 = MaxPooling2D(pool_size=(4, 4), strides=(1, 1), padding="valid", name="pool_2")(act_2)

conv_3 = Conv2D(32, kernel_size=(4, 4), name="conv_3")(pool_2)
act_3 = Activation("relu", name="act_3")(conv_3)
pool_3 = MaxPooling2D(pool_size=(8, 8), strides=(1, 1), padding="valid", name="pool_3")(act_3)

flat_1 = Flatten(name="flat_1")(pool_3)
dense_1 = Dense(128, activation="relu", name="dense_1")(flat_1)
batch_1 = BatchNormalization(name="batch_1")(dense_1)
dense_2 = Dense(64, activation="relu", name="dense_2")(batch_1)
batch_2 = BatchNormalization(name="batch_2")(dense_2)
dense_3 = Dense(32, activation="relu", name="dense_3")(batch_2)

isPothole = Dense(2, activation="softmax", name="pothole")(dense_3)

conv_4 = Conv2D(16, kernel_size=(4, 4), name="conv_4")(pool_3)
act_4 = Activation("relu", name="act_4")(conv_4)
pool_4 = MaxPooling2D(pool_size=(4, 4), strides=(1, 1), padding="valid", name="pool_4")(act_4)

conv_5 = Conv2D(16, kernel_size=(8, 8), name="conv_5")(pool_4)
act_5 = Activation("relu", name="act_5")(conv_5)
pool_5 = MaxPooling2D(pool_size=(4, 4), strides=(1, 1), padding="valid", name="pool_5")(act_5)

flat_2 = Flatten(name="flat_2")(pool_5)
dense_4 = Dense(128, activation="relu", name="dense_4")(flat_2)
batch_5 = BatchNormalization(name="batch_5")(dense_4)
add_2 = Add(name="add_2")([batch_1, batch_5])
batch_6 = BatchNormalization(name="batch_6")(add_2)
dense_5 = Dense(64, activation="relu", name="dense_5")(batch_6)
batch_7 = BatchNormalization(name="batch_7")(dense_5)
dense_6 = Dense(32, activation="relu", name="dense_6")(batch_7)
drop_2 = Dropout(rate=0.4, name="drop_2")(dense_6)
dense_7 = Dense(16, activation="relu", name="dense_7")(drop_2)
drop_3 = Dropout(rate=0.3, name="drop_3")(dense_7)

isSeverity = Dense(3, activation="softmax", name="severity")(drop_3)

model = Model(inputs=input_, outputs=[isPothole, isSeverity])

model.compile(
    optimizer='adam',
    loss={'pothole': 'categorical_crossentropy', 'severity': 'categorical_crossentropy'},
    metrics=['accuracy']
)

model.summary()
```
Defines and compiles the CNN model. 
## Training the Model 
```python
history = model.fit(
    my_training_batch_generator,
    steps_per_epoch=int(len(X_train_image_names) // batch_size),
    epochs=25,
    validation_data=my_validation_batch_generator,
    validation_steps=int(len(X_test_image_names) // batch_size),
    verbose=1
)
```
Trains the model using the custom data generators for a specified number of epochs.





























