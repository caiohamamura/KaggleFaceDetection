import numpy as np
import socket
import tensorflow as tf
from glob import glob
from tensorflow.examples.tutorials.mnist import input_data


def load_img(path):
    img_raw = tf.read_file(path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize_images(img, [600, 600])
    img /= 255.0
    img = tf.reshape(img, (1, 600, 600, 3))
    return img

def load_and_preprocess_from_path_label(path, label):
    label = tf.one_hot(tf.cast(label, tf.uint8), 4)
    return load_img(path), label


data_dir = 'data'

import random
all_image_paths = list(glob('%s/%s/%s/*' % (data_dir, 'real_and_fake_face', 'training_fake')))

random.shuffle(all_image_paths)
size = len(all_image_paths)
partition = int(size * 0.25)
images_train = all_image_paths[:-partition]
images_test = all_image_paths[-partition:]

label_names = [i[-8:-4] for i in images_train]
labels_train = [list(map(int,list(i))) for i in label_names]

label_names = [i[-8:-4] for i in images_test]
labels_test = [list(map(int,list(i))) for i in label_names]


all_image_real = list(glob('%s/%s/%s/**' % (data_dir, 'real_and_fake_face', 'training_real')))
size = len(all_image_real)
partition = int(size * 0.2)
images_train += all_image_real[:-partition]
images_test += all_image_real[-partition:]
import itertools
labels_train += itertools.repeat([0,0,0,0], size-partition)
labels_test += itertools.repeat([0,0,0,0], partition)



ds = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
ds = ds.map(load_and_preprocess_from_path_label)

ds_test = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
ds_test = ds_test.map(load_and_preprocess_from_path_label)


# ds_x = tf.data.Dataset.from_tensor_slices(all_image_paths)
# ds_x = ds_x.map(load_img)
# it_x = ds_x.make_one_shot_iterator()

# ds_y = tf.data.Dataset.from_tensor_slices(labels)
# it_y = ds_y.make_one_shot_iterator()

batch_size = 10
    

model = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(3, (3, 3), input_shape=(600, 600, 3)),
          tf.keras.layers.Activation('relu'),
          tf.keras.layers.MaxPooling2D((2,2)),

          tf.keras.layers.Conv2D(2, (3,3)),
          tf.keras.layers.Activation('relu'),
          tf.keras.layers.MaxPooling2D((2,2)),

          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(5),
          tf.keras.layers.Activation('relu'),
          tf.keras.layers.Dense(5),
          tf.keras.layers.Activation('relu'),
        #   tf.keras.layers.Dropout(rate=0.1),
          tf.keras.layers.Dense(4),
          tf.keras.layers.Activation('sigmoid')
        ])

model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.1, patience=10)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.fit(
        ds.make_one_shot_iterator(),
        steps_per_epoch=10,
        epochs=50,
        validation_data=ds_test.make_one_shot_iterator(),
        validation_steps=2,
        callbacks=[es, mc])

model.save_weights('first_try.h5') 