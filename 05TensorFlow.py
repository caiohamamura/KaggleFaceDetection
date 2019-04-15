import numpy as np
import socket
from glob import glob
import os
import random
from glob2 import glob
import itertools
from mpi4py import MPI
import math
import json
import tensorflow as tf

##########################################
## CONFIGURA PARAMETROS
###########################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # identificador da tarefa (rank)
jobs=json.load(open('saida.out'))

# rank = 0
# jobs = {'worker': ['s001-n094:2221'], 'ps': ['s001-n094:2220'], 'chief': ['s001-n095:2222']}

task={"index":0, "type": "ps"}
if rank == 1:
    task = {"index":0, "type": "chief"}
elif rank > 1:
    task = {"index":rank-2, "type": "worker"}
os.environ["TF_CONFIG"] = json.dumps({"cluster":jobs, "task":task})
run_config = tf.estimator.RunConfig()


PARTITION_TEST = 0.25
BATCH_SIZE = 10
NUM_CLASSES = 4


##########################################
## DEFINICAO DOS DADOS
###########################################


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


def train_fn(images_train, labels_train, epoch):
    ds_train = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
    ds_train = ds_train.apply(tf.data.experimental.map_and_batch(load_and_preprocess_from_path_label, BATCH_SIZE))
    ds_train = ds_train.repeat(epoch)
    ds_train.prefetch(tf.contrib.data.AUTOTUNE)
    iterator = ds_train.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

labels_train = [list(map(int,list(i))) for i in labels_train]


labels_test = [list(map(int,list(i))) for i in labels_test]
ds_test = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
ds_test = ds_test.apply(tf.data.experimental.map_and_batch(load_and_preprocess_from_path_label, BATCH_SIZE))
ds_test = ds_test.repeat()
ds_test.prefetch(tf.contrib.data.AUTOTUNE)
ds_test = ds_test.make_one_shot_iterator()

def get_next(dataset):
    features, labels = dataset.get_next()
    return features, labels


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    ds = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
    ds = ds.apply(tf.data.experimental.map_and_batch(load_and_preprocess_from_path_label, BATCH_SIZE))
    ds = ds.repeat()
    ds = ds.prefetch(tf.contrib.data.AUTOTUNE)
    return ds



#####################################
## MODELO
#####################################
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

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


cifar_est = tf.keras.estimator.model_to_estimator(keras_model=model
    ,config=run_config
    ,model_dir="kkt"
)

steps = math.ceil(len(labels_train)/BATCH_SIZE)


train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: train_fn(images_train, labels_train, steps),
    max_steps=steps)
eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: train_fn(images_test, labels_test, len(labels_test)/BATCH_SIZE/2),
    steps=len(labels_test)/BATCH_SIZE/2,
    start_delay_secs=0)
     
  # run !
tf.estimator.train_and_evaluate(
    cifar_est,
    train_spec,
    eval_spec
  )


# steps = range(math.ceil(len(labels_train)/BATCH_SIZE))
# for i in steps:
#     cifar_est.train(input_fn=lambda: train_fn(images_train, labels_train), steps=1)
#     cifar_est.test(input_fn=lambda: train_fn(images_test, labels_test), steps=len(labels_test)/BATCH_SIZE/2)