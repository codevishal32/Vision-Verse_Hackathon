import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import shutil
import random
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
print(tf.__version__)

#Converting the train dataset into test and train for evaluating
#3 images have been selected as test dataset
root_dir = 'vision-verse/data/train/'
classes_dir = ['crownandrootrot','healthywheat','leafrust','wheatloosesmut']
 
test_ratio = 0.0001


for cls in classes_dir:
    os.makedirs(root_dir +'train/' + cls)
    os.makedirs(root_dir +'test/' + cls)
    src = root_dir + cls
    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)* (1 - test_ratio))])
    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]
    for name in train_FileNames:
        shutil.copy(name, root_dir +'train/' + cls)
    for name in test_FileNames:
        shutil.copy(name, root_dir +'test/' + cls)
print("Copying Done!")
#Conversion is done



classes = ['crownandrootrot','healthywheat','leafrust','wheatloosesmut']

# loading training data
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        'vision-verse/data/train/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


#Understanding which number represents which class of wheat
class_dict = train_generator.class_indices
sample_count = train_generator.samples


# loading testing data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = train_datagen.flow_from_directory(
        'vision-verse/data/train/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


# initialising sequential model and adding layers to it
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(128, activation='relu'))
cnn.add(tf.keras.layers.Dense(64, activation='relu'))
cnn.add(tf.keras.layers.Dense(4, activation='softmax'))

# finally compile and train the cnn
cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
cnn.fit(x=train_generator, validation_data=test_generator, epochs=100)



#Predicting the results from the test dataset
def preprocess_image(path):
    img = load_img(path, target_size=(64, 64))
    a = img_to_array(img)
    a = np.expand_dims(a,axis = 0)
    a/=255.
    return a

test_images_dir = 'vision-verse/'
test_df = pd.read_csv('vision-verse/data/test.csv')

test_dfToList = test_df['path'].tolist()
test_ids = [str(item) for item in test_dfToList]

test_images = [test_images_dir + item for item in test_ids]

test_preprocessed_images = np.vstack([preprocess_image(fn) for fn in test_images])
np.save('test_preproc_CNN.npy', test_preprocessed_images)


array = cnn.predict(test_preprocessed_images, batch_size = 1, verbose = 1)
answer = np.argmax(array, axis = 1)
print(answer)


label = []

for value in answer:
    if value == 0:
        label.append('crownandrootrot')
    elif value == 1:
        label.append('healthywheat')
    elif value == 2:
        label.append('leafrust')
    else:
        label.append('wheatloosesmut')
        
      
#Converting the given columnd 'test_ids' and 'label' to csv    
import csv

with open('test.csv', 'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerows(zip(test_ids, label))








