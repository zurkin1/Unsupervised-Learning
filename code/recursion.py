#U2OS - bone cancer cells.
#RPE - retina pigment epithelium cells.
#HUVEC - human umblical vein endothelial cells.
#HEPG2 - human liver cancer cells.
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, concatenate, Input, add, Activation, Conv2D, MaxPooling2D, Flatten, ReLU
from tensorflow.keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.neighbors import NearestNeighbors
import umap
from matplotlib import pyplot as plt
import seaborn as sns
import multiprocessing
import random
import os
import time
import subprocess
#pip install tensorflow-gpu==2.0.0-beta1


AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16
NUM_IMAGES_TRAIN = 150000
NUM_IMAGES_TEST = 5000


def build_model(n_classes, input_shape=(224, 224, 3)):
    im_inp_1 = Input(shape=input_shape)
    im_inp_2 = Input(shape=input_shape)

    bn_1 = BatchNormalization()
    conv_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    ac_1 = ReLU()
    dr_1 = Dropout(0.2)

    bn_2 = BatchNormalization()
    conv_2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    ac_2 = ReLU()
    #mp_2 = MaxPooling2D(pool_size=(2,2), strides=(2, 2), padding = 'valid')
    dr_2 = Dropout(0.2)

    bn_3 = BatchNormalization()
    conv_3 = Conv2D(filters=4, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    ac_3 = ReLU()

    bn_4 = BatchNormalization()
    #gl_1 = GlobalAveragePooling2D()
    #dn_1 = Dense(512)
    #ac_4 = ReLU()

    #x1 = ac_6(dn_1(gl_1(bn_5(ac_5(conv_5(bn_4(ac_4(conv_4(bn_3(ac_3(conv_3(dr_2(bn_2(mp_2(ac_2(conv_2(dr_1(bn_1(ac_1(conv_1(im_inp_1)))))))))))))))))))))
    x1 = bn_4(ac_3(conv_3(bn_3(dr_2(ac_2(conv_2(bn_2(dr_1(ac_1(conv_1(bn_1(im_inp_1))))))))))))
    x2 = bn_4(ac_3(conv_3(bn_3(dr_2(ac_2(conv_2(bn_2(dr_1(ac_1(conv_1(bn_1(im_inp_2))))))))))))

    x = add([x1, x2])
    x = MaxPooling2D(pool_size=(3,3))(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)
    out = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=[im_inp_1, im_inp_2], outputs=out)
    model.compile(SGD(0.001, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def read_tfrecord(example):
    features = {
        "image1": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "image2": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)
    image1 = tf.image.decode_jpeg(example['image1'], channels=3)
    image1 = tf.cast(image1, tf.float32) / 255.0
    image1 = tf.reshape(image1, [100, 100, 3])
    image2 = tf.image.decode_jpeg(example['image2'], channels=3)
    image2 = tf.cast(image2, tf.float32) / 255.0
    image2 = tf.reshape(image2, [100, 100, 3])
    class_label = tf.one_hot(tf.cast(example['class'], tf.int32), 4, dtype=tf.int32)
    #class_label = tf.cast(example['class'], tf.int32)
    return (image1, image2), class_label


def get_batched_dataset(mode, shuffle=True):
    BASE_PATH = './' + mode + '/'
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = not shuffle
    dataset = tf.data.Dataset.list_files(BASE_PATH+'*.*', shuffle=shuffle)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    #dataset = dataset.cache()  # This dataset fits in RAM
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)

    return dataset


def train_patches_model():
    model = build_model(input_shape=(100, 100, 3), n_classes=4)
    #model = tf.keras.models.load_model('models/patch_model.h5')
    #model.compile(SGD(0.0001, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy']) #Adam(0.0001)
    model.summary()

    #Train.
    checkpoint = ModelCheckpoint('model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    history = model.fit(get_batched_dataset('train'),
                        steps_per_epoch=10, #NUM_IMAGES_TRAIN//BATCH_SIZE
                        validation_data=get_batched_dataset('test'),
                        validation_steps=10, #NUM_IMAGES_TEST//BATCH_SIZE
                        callbacks=[checkpoint],
                        #use_multiprocessing=True,
                        #workers=multiprocessing.cpu_count()-1,
                        verbose=1,
                        epochs=500)

    #with open('history.json', 'w') as f:
    #    json.dump(history.history, f)
    print(history.history)
    return model


def calculate_TSNE_UMAP():
    patch_model = tf.keras.models.load_model('data/model.h5')
    layer_output = tf.keras.layers.Flatten()(patch_model.layers[-16].output)
    new_model = Model(inputs = patch_model.input, outputs = layer_output)
    new_model.summary()
    #plot_model(new_model, to_file='new_model.png', show_shapes=True)
    #plt.show()
    input_array = new_model.predict_generator(get_batched_dataset('test', shuffle=False), steps=63, verbose=1) #NUM_IMAGES_TEST//BATCH_SIZE
    labels = np.load('data/labels.npy')
    print(time.ctime(), ' TSNE/UMAP...')
    #_model = PCA(2)
    _model = FastICA(2, max_iter=3000, tol=0.00001, fun='cube') #
    #_model = TSNE(n_components=2, verbose=1, perplexity=60, n_iter=2000, metric='hamming') #method='exact'
    #_model = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=2000, method='exact')
    #_model = umap.UMAP(n_components=2, n_neighbors=5)
    _results = _model.fit_transform(input_array)
    #clusters = DBSCAN(eps=0.5, min_samples=5).fit(input_array)
    #clusters = SpectralClustering(n_clusters=4).fit(input_array) #, assign_labels='discretize', n_init=100
    #cluster_labels = clusters.labels_

    #Check how dense are the clusters
    nn = NearestNeighbors(n_neighbors=5)
    nbrs = nn.fit(_results)
    distances, indices = nbrs.kneighbors(_results)
    f = lambda x: labels[x]
    a = f(indices)
    #a_match = a.T == a.T[0,:] #Check that classes in each row are in agreement.
    #a_all = np.sum(np.all(a_match, axis=0)) #n=1, 377s
    a_count = np.apply_along_axis(lambda x: np.bincount(x, minlength=4), axis=1, arr=a.astype(int))
    a_max = np.argmax(a_count, axis=1)
    sum_equal = (a_max == labels[0:len(a_max)]).sum()
    print(f'Number of points matching their neighbors: {sum_equal} out of {len(a_max)}') #699, 1008

    print(time.ctime(), ' Print results...')
    _df = pd.DataFrame({'X':_results[:,0], 'Y':_results[:,1], 'label':labels[:len(_results[:,0])]}, index=range(len(_results[:,0])))
    plt.figure(figsize=(16,10))
    sns_plot = sns.scatterplot(x='X', y='Y', data=_df, palette = ['purple', 'red', 'orange', 'green'], legend="brief", alpha=0.3, hue="label") #palette=sns.color_palette("hls", 10), sns.cubehelix_palette(as_cmap=True)
    fig = sns_plot.get_figure()
    plt.show()
    #fig.savefig("model_output.png")

def train_autoencoder_model():
    print('Use combined model to predict cell type. Create inputs...')
    patch_model = tf.keras.models.load_model('data/model.h5')
    layer = patch_model.layers[-4].output #tf.keras.layers.Flatten()
    patch_model = Model(inputs=patch_model.input, outputs=layer)
    input_for_autoencoder = patch_model.predict_generator(get_batched_dataset('train', shuffle=False), steps=NUM_IMAGES_TRAIN/BATCH_SIZE, verbose=1)
    np.save('data/input_for_autoencoder.npy', input_for_autoencoder)
    #input_for_autoencoder = np.load('data/input_for_autoencoder.npy')

    print('Train autoencoder model...')
    input_layer = Input(16)
    x = BatchNormalization()(Dense(16, activation='relu')(input_layer))
    x = BatchNormalization()(Dense(4, activation='relu')(x))
    x = BatchNormalization()(Dense(2, activation='relu')(x))
    x = BatchNormalization()(Dense(4, activation='relu')(x))
    x = BatchNormalization()(Dense(16, activation='relu')(x))
    #x = BatchNormalization()(Dense(36, activation='relu')(x))
    autoencoder = Model(input_layer, x)
    autoencoder.summary()
    autoencoder.compile(SGD(0.0001, nesterov=True), loss='mean_squared_error')
    x_train = input_for_autoencoder[0: int(0.9*input_for_autoencoder.shape[0])]
    x_test = input_for_autoencoder[int(0.9*input_for_autoencoder.shape[0]):]
    checkpoint = ModelCheckpoint('autoencoder.h5', monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto')
    autoencoder.fit(x_train, x_train, batch_size=10, epochs=50, validation_data=(x_test,x_test), callbacks=[checkpoint], verbose=1)

    #Use only the first half of the model.
    print('Use combined models...')
    autoencoder = tf.keras.models.load_model('autoencoder.h5')
    layer = autoencoder.layers[-3].output
    autoencoder = Model(inputs = autoencoder.input, outputs = layer)
    newOutputs = autoencoder(patch_model.output)
    newModel = Model(patch_model.input, newOutputs)
    _results = newModel.predict_generator(get_batched_dataset('test', shuffle=False), steps=20, verbose=1)

    #Plot results.
    labels = np.load('data/labels.npy')
    _df = pd.DataFrame({'X':_results[:320,0], 'Y':_results[:320,1], 'label':labels[:320]}, index=range(320))
    plt.figure(figsize=(16,10))
    sns_plot = sns.scatterplot(x='X', y='Y', data=_df, palette = ['purple', 'red', 'orange', 'green'], legend="brief", alpha=0.3, hue="label") #palette=sns.color_palette("hls", 10), sns.cubehelix_palette(as_cmap=True)
    fig = sns_plot.get_figure()
    fig.savefig("autoencoder_output.png")
    #subprocess.run(['gdrive', 'upload', 'autoencoder_output.png'])


    #files = os.listdir('./data3')
    #random.shuffle(files)
    #i = 0
    #for image in files:
    #    image_bits = tf.io.read_file('./data3/' + image)
    #    decoded = tf.image.decode_jpeg(image_bits)
        #output_model_A = new_model.predict(([decoded], [decoded]))
        #output_model_A = minmax_scale(output_model_A)
        #output_model_A = (output_model_A - np.min(output_model_A))/np.max(output_model_A)
        #output_model_B = decoder_model2.predict(output_model_A)
    #    pred = newModel.predict(([decoded], [decoded]))
    #    print(image[0:5], ' ', pred)
    #    i += 1
    #    if i == 10:
    #        break


def printTFRecord():
    print('Get the lables from TFRecords. Labels are cell types not patch location...')
    #tf.enable_eager_execution()
    #image_, classes_t = dataset.make_one_shot_iterator().get_next()
    #with tf.Session() as sess:
    #    while True:
    #        image, classes = sess.run([image_t, classes_t])
    #        break
    label_array = np.empty(shape=(0))
    """
    for example in tf.data.TFRecordDataset("C:/Temp/test_15.tfrec"): #tf.python_io.tf_record_iterator
        #test = tf.train.Example.FromString(example)
        #print(test)
        #print(test.features.feature['image2'].ByteSize()) # 53075
        parsed_features = read_tfrecord(example) #tf.parse_single_example(example, features)
        #image1 = parsed_features[0]
        #x1 = tf.image.decode_jpeg(image1[0][0], channels=3)
        #x2 = tf.reshape(x1, [100, 100, 3])
        #x3 = tf.cast(x2, tf.float32) / 255.0
        label = parsed_features[1]

        #sess = tf.Session() # Tensorflow 1.4
        #with sess.as_default():
            #x3.eval()
            #print(label.eval())
        label_array = np.concatenate([label_array, [label]])
        #tf.train.Example.FromString(image2)
    """
    data = get_batched_dataset('test', shuffle=False)
    i = 0
    for (x, y), z in data:
        if i == 63:
            break
        print('\r', i, end='')
        z1 = [np.where(r == 1)[0][0] for r in z]
        label_array = np.concatenate([label_array, z1])
        i += 1
    np.save('data/labels.npy', label_array)


if __name__ == '__main__':
    #train_patches_model()
    calculate_TSNE_UMAP()
    #train_autoencoder_model()
    #printTFRecord()