import numpy as np
from PIL import Image
import tensorflow as tf
import json
import keras
from keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras import Model #, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, concatenate, Input, add, Activation, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
#!pip install --upgrade scikit-image
import multiprocessing
import image_slicer
import random
import os
import io

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytestring_feature(list_of_bytestrings):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[list_of_bytestrings]))

def _int_feature(list_of_ints): # int64
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats): # float32
  return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def read_image_and_label(img_path):
    tiles = list(image_slicer.slice(img_path, 9, save=False))
    center = tiles[4]
    tiles.remove(center)
    item = random.choice(tiles)
    label = item.number
    # Convert the image to raw bytes.
    #center = np.array(center.image).tostring()
    #item = np.array(item.image).tostring()

    return center.image, item.image, label


def to_tfrecord(img_bytes1, label): #img_bytes2,
    feature = {
        "image1": _bytestring_feature(img_bytes1),  # one image in the list
        #"image2": _bytestring_feature([img_bytes2]),  # one image in the list
        "class": _int_feature([label])  # one class in the list
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def convert2(mode, out_path, base_path):
    dataset = tf.data.Dataset.list_files(base_path + mode, seed=10000) #This also shuffles the files.
    dataset = tf.data.Dataset.list_files('C:/Temp/recursion2/test/', seed=10000)  # This also shuffles the files.
    dataset = dataset.map(read_image_and_label)
    with tf.io.TFRecordWriter(out_path) as out_file:
        for image1, image2, label in enumerate(dataset):
            example = to_tfrecord(image1.numpy(), image2.numpy(), label.numpy())
            out_file.write(example.SerializeToString())
        print("Wrote file records")


def convert_shard(image_files, mode, out_path, base_path, shard):
    # Open a TFRecordWriter for the output-file.
    with tf.io.TFRecordWriter(out_path + 'stam' + str(shard)) as writer:
        for image in image_files:
                # Iterate over all the image-paths and class-labels.
                #print('\r', f"{i / num_images * 100:.0f}%", end='')
                #tiles = list(image_slicer.slice(base_path + mode + '/' + image, 9, save=False))
                #center = tiles[4]
                #tiles.remove(center)
                #item = random.choice(tiles)
                #label = item.number

                bits = Image.open(base_path + mode + '/' + image)
                #bits = tf.io.read_file(image)
                #image = tf.image.decode_jpeg(bits)
                b = io.BytesIO()
                bits.save(b, 'jpeg')
                image = b.getvalue()
                label = 9

                # Convert the image to raw bytes.
                # center = np.array(center.image).tostring()
                # item = np.array(item.image).tostring()
                # Create a dict with the data we want to save in the
                # TFRecords file. You can add more relevant data here.
                ##data = \
                ##    {
                ##        'image1': wrap_bytes(center),
                ##        'image2': wrap_bytes(item),
                ##        'label': wrap_int64(label)
                ##    }
                # Wrap the data as TensorFlow Features.
                ##feature = tf.train.Features(feature=data)
                # Wrap again as a TensorFlow Example.
                # example = tf.train.Example(features=feature)

                #example = to_tfrecord(np.array(center.image).tostring(), np.array(item.image).tostring(), label)
                #example = to_tfrecord(np.array(center.image).tostring(), label)
                example = to_tfrecord(image, label)

                # Serialize the data.
                serialized = example.SerializeToString()
                # Write the serialized data to the TFRecords file.
                writer.write(serialized)


def convert(mode, out_path, base_path):
    print("Converting: " + out_path)
    image_files = [file for file in os.listdir(base_path + mode)]
    # Number of images. Used when printing the progress.
    num_images = len(image_files)
    i = 0
    for j in range(num_images//1000):
        convert_shard(image_files[j*1000:(j+1)*1000], mode, out_path, base_path, i)
        i += 1


BASE_PATH='C:/temp/recursion2/'
BATCH_SIZE = 16
NUM_IMAGES_TRAIN = 73030
NUM_IMAGES_TEST = 39794


def build_model(n_classes, input_shape=(224, 224, 3)):
    im_inp_1 = Input(shape=input_shape)
    im_inp_2 = Input(shape=input_shape)

    conv_1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu')
    mp_1 = MaxPooling2D(pool_size=(2,2), strides=(2, 2), padding = 'valid')
    bn_1 = BatchNormalization()
    dr_1 = Dropout(0.2)

    conv_2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu') #11
    mp_2 = MaxPooling2D(pool_size=(2,2), strides=(2, 2), padding = 'valid')
    bn_2 = BatchNormalization()
    dr_2 = Dropout(0.2)

    conv_3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu') #5
    bn_3 = BatchNormalization()

    #conv_4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
    #bn_4 = BatchNormalization()

    #conv_5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
    #mp_5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
    #bn_5 = BatchNormalization()

    gl_1 = GlobalAveragePooling2D()
    fl_1 = Flatten()
    dn_1 = Dense(4096, input_shape=(224 * 224 * 3,), activation='relu')


    #x1 = dn_1(fl_1(bn_5(mp_5(conv_5(bn_4(conv_4(bn_3(conv_3(bn_2(mp_2(conv_2(bn_1(mp_1(conv_1(im_inp_1)))))))))))))))
    x1 = dn_1(gl_1(bn_3(conv_3(dr_2(bn_2(mp_2(conv_2(dr_1(bn_1(mp_1(conv_1(im_inp_1))))))))))))
    x2 = dn_1(gl_1(bn_3(conv_3(dr_2(bn_2(mp_2(conv_2(dr_1(bn_1(mp_1(conv_1(im_inp_2))))))))))))

    #x1 = GlobalAveragePooling2D()(x1)
    #x2 = GlobalAveragePooling2D()(x2)

    x = add([x1, x2])
    x = Dense(2048, activation='relu')(x)
    #x = Dropout(0.4)(x)
    x = Dense(1024, activation='relu')(x)
    #x = Dropout(0.4)(x)
    out = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=[im_inp_1, im_inp_2], outputs=out)
    model.compile(Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def _parse_function(proto):
    features = {
        'image1': tf.FixedLenSequenceFeature([133 * 133 * 3], tf.string, allow_missing=True),
        #'image2': tf.FixedLenSequenceFeature([133 * 133 * 3], tf.string, allow_missing=True),
        'label': tf.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)
    }
    parsed_features = tf.parse_single_example(proto, features)

    image1 = parsed_features['image1']
    image1 = tf.image.decode_jpeg(image1[0][0], channels=3)
    image1 = tf.cast(image1, tf.float32) / 255.0
    image1 = tf.reshape(image1, [133, 133, 3])

    image2 = parsed_features['image2']
    image2 = tf.image.decode_jpeg(image2[0][0], channels=3)
    image2 = tf.cast(image2, tf.float32) / 255.0
    image2 = tf.reshape(image2, [133, 133, 3])

    #image1 = tf.cast(parsed_features['image1'], tf.float32)
    #image2 = tf.cast(parsed_features['image2'], tf.float32)
    #image1 = tf.reshape(image1 / 255, [133, 133, 3])
    #image2 = tf.reshape(image2 / 255, [133, 133, 3])

    y = parsed_features['label'] #tf.cast(parsed_features['label'], tf.int64)

    return (image1, image2), y


def load_dataset(input_path, batch_size):
    dataset = tf.data.TFRecordDataset(input_path)
    dataset = dataset.map(_parse_function, num_parallel_calls=16)
    dataset = dataset.shuffle(10000).repeat()  # shuffle and repeat
    dataset = dataset.batch(batch_size).prefetch(1)  # batch and prefetch

    return dataset #dataset.make_one_shot_iterator()


def load_batched_dataset(mode):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files('C:/Temp/recursion2a/'+mode+'.*')
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=16)
    dataset = dataset.map(_parse_function, num_parallel_calls=16)

    dataset = dataset.cache()  # This dataset fits in RAM
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(1)  #

    return dataset

if __name__ == '__main__':
    #convert('train', 'C:/Temp/recursion2/train.tf', BASE_PATH)
    convert('test', 'C:/Temp/recursion2/test.tf', BASE_PATH)
"""
    train_iterator = load_batched_dataset('train') #load_dataset(BASE_PATH+'train.tf', BATCH_SIZE)
    val_iterator = load_batched_dataset('test') #load_dataset(BASE_PATH+'test.tf', BATCH_SIZE)

    model = build_model(input_shape=(133, 133, 3), n_classes=8)
    #model = load_model('model.h5')
    model.summary()

    #Train.
    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    history = model.fit(train_iterator,
                        steps_per_epoch=NUM_IMAGES_TRAIN//BATCH_SIZE,
                        validation_data=val_iterator,
                        validation_steps=NUM_IMAGES_TEST//BATCH_SIZE,
                        #callbacks=[checkpoint],
                        #use_multiprocessing=True,
                        #workers=multiprocessing.cpu_count()-1,
                        verbose=1,
                        epochs=10)

    with open('history.json', 'w') as f:
        json.dump(history.history, f)
"""
    #image_, classes_t = dataset.make_one_shot_iterator().get_next()
    #with tf.Session() as sess:
    #    while True:
    #        image, classes = sess.run([image_t, classes_t])
    #        break
    for example in tf.python_io.tf_record_iterator("C:/Temp/recursion2/test.tfstam0"):
        #test = tf.train.Example.FromString(example)
        #print(test)
        #print(test.features.feature['image2'].ByteSize()) # 53075
        parsed_features = tf.parse_single_example(example, features)
        #print(parsed_features['image2'])
        image1 = parsed_features['image1']
        x1 = tf.image.decode_jpeg(image1[0][0], channels=3)
        x2 = tf.reshape(x1, [133, 133, 3])
        x3 = tf.cast(x2, tf.float32) / 255.0
        sess = tf.Session()
        with sess.as_default():
            #x3.eval()
            x3.eval()
        #tf.train.Example.FromString(image2)
        break