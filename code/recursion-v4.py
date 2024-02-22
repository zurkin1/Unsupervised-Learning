import json
import numpy as np
import keras
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, concatenate, Input, add, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
#!pip install --upgrade scikit-image
import multiprocessing
import image_slicer
import random
import os
from tqdm import tqdm


BATCH_SIZE = 16
BASE_PATH = 'C:/Temp/recursion400x400'
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

if __name__ == '__main__':
    #Preprocessing.
    #slice9()


    def generator_multiple(generator, mode, target_size=(133,133)):
        genX1 = generator.flow_from_directory(BASE_PATH+'/'+mode+'_center', target_size=target_size, batch_size=BATCH_SIZE, shuffle=False)
        genX2 = generator.flow_from_directory(BASE_PATH+'/'+mode+'_random', target_size=target_size, batch_size=BATCH_SIZE, shuffle=False)
        while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[0]], X2i[1] #Yield both images and theor mutual label.

    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()
    traingenerator = generator_multiple(generator=train_datagen, mode='train')
    valgenerator = generator_multiple(generator=val_datagen, mode='test')
    model = build_model(input_shape=(133, 133, 3), n_classes=8)
    #model = load_model('model.h5')
    model.summary()

    #Train.
    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    history = model.fit_generator(traingenerator,
                                  steps_per_epoch=NUM_IMAGES_TRAIN/BATCH_SIZE,
                                  validation_data=valgenerator,
                                  validation_steps=NUM_IMAGES_TEST/BATCH_SIZE,
                                  #callbacks=[checkpoint],
                                  #use_multiprocessing=True,
                                  #workers=multiprocessing.cpu_count()-1,
                                  verbose=1,
                                  epochs=10)

    with open('history.json', 'w') as f:
        json.dump(history.history, f)