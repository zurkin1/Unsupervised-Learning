import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from numpy import expand_dims
import math
#!pip install tensorflow==2.0.0-beta1
#print(tf.__version__)
import image_slicer
import random
import os
from matplotlib import pyplot


AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API.
PATCHES_CLASSES = [b'2', b'4', b'6', b'8']
CELLTYPES_CLASSES = [b'1', b'2', b'3', b'4']
#CLASSES = [-3, -1, 1, 3]


def slice9():
  image_files = [file for file in os.listdir('C:/Temp/recursion/')]
  random.shuffle(image_files)
  j = 0
  for image in image_files:
      print('\r', f"{j / len(image_files) * 100:.0f}%", end='')
      tiles = list(image_slicer.slice('C:/Temp/recursion/'+image, 9, save=False))
      center = tiles[4]
      tiles.remove(tiles[8])
      tiles.remove(tiles[6])
      tiles.remove(center)
      tiles.remove(tiles[2])
      tiles.remove(tiles[0])
      item=random.choice(tiles)
      center_image = center.image.crop((15, 15, 115, 115))
      item_image = item.image.crop((15, 15, 115, 115))
      center_image.save('C:/Temp/center/'+str(item.number)+'.'+image)
      item_image.save('C:/Temp/random/'+str(item.number)+'.'+image)
      j += 1


def read_image_and_label(img_path):
    bits1 = tf.io.read_file(img_path)
    image1 = tf.image.decode_jpeg(bits1)
    img_path2 = 'C:/Temp/5_test_labels/random/' + tf.strings.split(img_path, sep='\\')[-1] #random
    bits2 = tf.io.read_file(img_path2)
    image2 = tf.image.decode_jpeg(bits2)
    #center = tf.slice(image1, [132, 132, 0], [133, 133, 3])
    #sliceX = (-1 if tf.random.uniform(shape=(0,1), minval=0, maxval=2, dtype=tf.dtypes.int32) == 0 else 1) # random.randint(0,1)
    #sliceY = (-1 if tf.random.uniform(shape=(0,1), minval=0, maxval=2, dtype=tf.dtypes.int32) == 0 else 1)
    #rand = tf.slice(image1, [133+sliceX*133, 133+sliceY*133, 0], [133, 133, 3])
    #label = sliceX + 2 * sliceY
    label = tf.strings.split(img_path, sep='\\') # train/8.U2OS-05_4_O22_s1.jpeg
    label = tf.strings.split(label[-1], sep='.')
    return image1, image2, label[0] # center, rand, label


def recompress_image(image1, image2, label):
  image1 = tf.cast(image1, tf.uint8)
  image1 = tf.image.encode_jpeg(image1, optimize_size=True, chroma_downsampling=False)
  image2 = tf.cast(image2, tf.uint8)
  image2 = tf.image.encode_jpeg(image2, optimize_size=True, chroma_downsampling=False)
  return image1,image2,label


def _bytestring_feature(list_of_bytestrings):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): # int64
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats): # float32
  return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def to_tfrecord(img_bytes1, img_bytes2, label):
  #class_num = np.argmax(np.array(CLASSES)==label)
  class_num = CELLTYPES_CLASSES.index(label)
  feature = {
      "image1": _bytestring_feature([img_bytes1]), # one image in the list
      "image2": _bytestring_feature([img_bytes2]),
      "class": _int_feature([class_num]),        # one class in the list
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def print_layers():
    model = tf.keras.models.load_model('data/model.h5')
    filters, biases = model.layers[3].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters, ix = 6, 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = pyplot.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()

    # summarize feature map shapes
    for i in range(len(model.layers)):
        layer = model.layers[i]
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        # summarize output shape
        print(i, layer.name, layer.output.shape) #3, 7, 11
    model2 = Model(inputs=model.inputs, outputs=model.layers[11].output)
    img = load_img('data3/6.U2OS-05_4_G08_s2.jpeg')
    img = img_to_array(img)
    img = expand_dims(img, axis=0)
    feature_maps = model2.predict((img, img))
    # plot all 64 maps in an 8x8 squares
    square = 2 #8 4
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()


if __name__ == '__main__':
    #slice9()

    #tf.enable_eager_execution()
    dataset = tf.data.Dataset.list_files('C:/Temp/5_test_labels/center/' + '*.jpeg', seed=10000)  # This also shuffles the images
    dataset = dataset.map(read_image_and_label)
    dataset = dataset.map(recompress_image, num_parallel_calls=AUTO)

    # Split to shards.
    #number_samples = 40000
    #SHARDS = 8
    #shard_size = math.ceil(1.0 * number_samples / SHARDS)
    shard_size = 1000 #5000
    print('Num samples per shard', shard_size)
    dataset = dataset.batch(shard_size)

    for shard, (image1, image2, label) in enumerate(dataset):
        # shard_size = image.numpy().shape[0]
        filename = "C:/Temp/test_labels_" + "{:02d}.tfrec".format(shard)

        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(shard_size-10):
                example = to_tfrecord(image1.numpy()[i], image2.numpy()[i], label.numpy()[i])
                out_file.write(example.SerializeToString())
            print("Wrote file {} containing {} records".format(filename, shard_size))
"""
import tensorflow as tf
# with tf.Session():
#      bits1 = tf.io.read_file('data/HEPG2-01_1_B03_s1.jpeg')
#      image1 = tf.image.decode_jpeg(bits1)
#      center = tf.slice(image1, [0, 132, 0], [133, 133, 3])
#      tmpimage = tf.cast(center, tf.uint8)
#      tmpimage = tf.image.encode_jpeg(tmpimage, optimize_size=True, chroma_downsampling=False)
#      tf.write_file('C:/Temp/test-2.jpeg', tmpimage).run()
"""