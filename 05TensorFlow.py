import numpy as np
import socket
import tensorflow as tf
from glob import glob
import os
import random
from glob2 import glob
import itertools
from tensorflow.keras import backend as K
import horovod.keras as hvd
hvd.init()
rank = str(hvd.local_rank())
rank = str(1)
PARTITION_TEST = 0.25
BATCH_SIZE = 20
NUM_CLASSES = 4



def config_proto():
    """Returns session config proto.
    Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
    """
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = 12
    config.experimental.collective_group_leader = '/job:worker/replica:0/task:0'
    config.device_count['CPU'] = 12
    config.gpu_options.visible_device_list = rank
    # For collective_all_reduce, ignore all devices except current worker.
    if rank == 0:
        config.device_filters.append(
            '/job:%s/replica:0/task:%s' % ("ps", rank))
    else:
        config.device_filters.append(
            '/job:%s/replica:0/task:%s' % ("worker", rank))

    return config

# Carrega tensor com base no caminho da imagem
def load_img(path):
    img_raw = tf.read_file(path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize_images(img, [600, 600])
    img /= 255.0
    img = tf.reshape(img, (600, 600, 3))
    return img

# Carrega imagem e label no padrão tensorflow
def load_and_preprocess_from_path_label(path, label):
    label = tf.cast(label, tf.uint8)
    return load_img(path), label

# Local dos arquivos de dados
data_dir = 'data'
training_fake_path = "%s/real_and_fake_face/training_fake" % data_dir

# Renomeia o arquivo que veio com o nome errado
for i in glob(r"%s\%s" % (training_fake_path, "easy_116_111.jpg")): os.rename(i, os.path.join(training_fake_path, 'easy_116_1111.jpg'))

# Carrega caminho para todos os fake
all_fake = glob(os.path.join(data_dir, 'real_and_fake_face', 'training_fake', '*'))

#Obtem labels e transforma em binario
label_fake = [i[-8:-4] for i in all_fake]

all_real =  glob(os.path.join(data_dir, 'real_and_fake_face', 'training_real', '*'))
label_real = ['0000']*len(all_real)

all_image_paths = all_fake + all_real
all_labels = label_fake + label_real


from sklearn.model_selection import train_test_split
#Particiona em train e test estratificado, 
# mantendo proporcao das classes
images_train,images_test,labels_train, labels_test = train_test_split(
        all_image_paths, all_labels, 
        stratify=all_labels,
        test_size = PARTITION_TEST)



labels_train = [list(map(int,list(i))) for i in labels_train]
ds = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
ds = ds.apply(tf.data.experimental.map_and_batch(load_and_preprocess_from_path_label, BATCH_SIZE))
ds = ds.repeat()
ds = ds.prefetch(tf.contrib.data.AUTOTUNE)

labels_test = [list(map(int,list(i))) for i in labels_test]
ds_test = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
ds_test = ds_test.apply(tf.data.experimental.map_and_batch(load_and_preprocess_from_path_label, BATCH_SIZE))
ds_test = ds_test.repeat()
ds_test.prefetch(tf.contrib.data.AUTOTUNE)

# ds_x = tf.data.Dataset.from_tensor_slices(all_image_paths)
# ds_x = ds_x.map(load_img)
# it_x = ds_x.make_one_shot_iterator()

# ds_y = tf.data.Dataset.from_tensor_slices(labels)
# it_y = ds_y.make_one_shot_iterator()

conf = config_proto()
epochs = int(math.ceil(50.0 / hvd.size()))
K.set_session(tf.Session(config=config))

model = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(32, (3, 3), input_shape=(600, 600, 3)),
          tf.keras.layers.Activation('relu'),
          tf.keras.layers.MaxPooling2D((2,2)),

          tf.keras.layers.Conv2D(32, (3,3)),
          tf.keras.layers.Activation('relu'),
          tf.keras.layers.MaxPooling2D((2,2)),

          tf.keras.layers.Conv2D(64, (3,3)),
          tf.keras.layers.Activation('relu'),
          tf.keras.layers.MaxPooling2D((2,2)),

          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(60),
          tf.keras.layers.Activation('relu'),
          tf.keras.layers.Dropout(rate=0.5),
          tf.keras.layers.Dense(NUM_CLASSES),
          tf.keras.layers.Activation('sigmoid')
        ])


from tensorflow.keras import optimizers

opt = optimizers.Adadelta(1.0 * hvd.size())
opt = hvd.DistributedOptimizer(opt)

sgd = optimizers.SGD(lr=1, decay=.3, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
        # optimizer=sgd,
        optimizer="rmsprop",
        metrics=['acc'])



from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.1, patience=10)
callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0)
]

if hvd.rank() == 0:
    callbacks.append(
        ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    )
import math

model.fit(
        ds.make_one_shot_iterator(),
        steps_per_epoch=int(math.ceil(len(label_fake)/BATCH_SIZE)),
        batch_size=BATCH_SIZE,
        epochs=50,
        validation_data=ds_test.make_one_shot_iterator(),
        validation_steps=int(math.ceil(len(labels_test)/BATCH_SIZE)),
        callbacks=callbacks)

x_test, y_test = ds_test.make_one_shot_iterator().get_next()
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




