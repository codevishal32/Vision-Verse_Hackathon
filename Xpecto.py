import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import shutil
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf 
import csv
root_dir = 'data/train/'
classes_dir = ['crownandrootrot','healthywheat','leafrust','wheatloosesmut']
 
test_ratio = 0.0001


# =============================================================================
# for c in classes_dir:
#     os.makedirs(root_dir +'train/' + c)
#     os.makedirs(root_dir +'test/' + c)
#     src = root_dir + c
#     allFileNames = os.listdir(src)
#     np.random.shuffle(allFileNames)
#     train_FileNames, test_FileNames = np.split(np.array(allFileNames),
#                                                           [int(len(allFileNames)* (1 - test_ratio))])
#     train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
#     test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]
#     for name in train_FileNames:
#         shutil.copy(name, root_dir +'train/' + c)
#     for name in test_FileNames:
#         shutil.copy(name, root_dir +'test/' + c)
# 
# =============================================================================



classes = ['crownandrootrot','healthywheat','leafrust','wheatloosesmut']


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)




test_datagen = ImageDataGenerator(rescale=1./255)
tstg = train_datagen.flow_from_directory(
        'data/train/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
tg = train_datagen.flow_from_directory(
        'data/train/train',
        target_size=(64, 64),
        batch_size=30,
        class_mode='categorical')

cd = tg.class_indices


cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=1, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=1, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=1, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(256, activation='relu'))
cnn.add(tf.keras.layers.Dense(64, activation='relu'))
cnn.add(tf.keras.layers.Dense(64, activation='relu'))
cnn.add(tf.keras.layers.Dense(4, activation='softmax'))


cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
cnn.fit(tg, validation_data=tstg, epochs=1000)

def preprocess_image(path):
    img = load_img(path, target_size=(64, 64))
    a = img_to_array(img)
    a = np.expand_dims(a,axis = 0)
    a/=255.
    return a

test_df = pd.read_csv('data/test.csv')

test_dfToList = test_df['path'].tolist()
test_ids = [str(item) for item in test_dfToList]

test_images = [item for item in test_ids]

test_preprocessed_images = np.vstack([preprocess_image(fn) for fn in test_images])


array = cnn.predict(test_preprocessed_images, batch_size = 1, verbose = 1)
a = np.argmax(array, axis = 1)
print(a)


lb = []

for value in a:
    if value == 0:
        lb.append('crownandrootrot')
    elif value == 1:
        lb.append('healthywheat')
    elif value == 2:
        lb.append('leafrust')
    else:
        lb.append('wheatloosesmut')
        
import csv

with open('test.csv', 'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerows(zip(test_ids, lb))








