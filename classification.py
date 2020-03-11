import os
import time
from skimage.transform import resize
import numpy as np
from morph import dilate, erode, adapt
import keras
from keras.datasets import mnist
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Conv2D, Activation, Add, BatchNormalization
from keras import optimizers
from keras.models import Model
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import skimage.io as io
from grayscale_morph import gray_dilate, gray_erode, sliding_window
import argparse
from utils import *
from model import *
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.layers import Input, Activation, MaxPooling2D, Flatten, Dense, Dropout, Conv2D, Concatenate
from keras.callbacks import TensorBoard

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int,default=100)
parser.add_argument('--batch_size', type=int,default=64)
parser.add_argument('--gpu_id', type=int, default=3)
parser.add_argument('--height', type=int, default=32)
parser.add_argument('--width', type=int, default=32)
parser.add_argument('--model_name', type = str, default = "1128_res_bn_dropout_cifar10")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

height = args.height
width = args.width

import cv2
def res_mnn(input_shape= (height, width, 3)):
    inputs = Input(shape=(height, width, 3))

    x1 = residue_block(inputs, filters=16, filter_size=(3,3), strides=(1,1))
    x1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x1)
    x1 = MaxPooling2D(pool_size=(2,2))(x1)



    #x2 = residue_block(x1, filters = 16, filter_size=(3,3), strides=(1,1))
    #x2 = MaxPooling2D(pool_size=(2,2))(x2)

    #x3 = residue_block(x2, filters=16, filter_size=(3,3), strides=(1,1))
    ##x3 = MaxPooling2D(pool_size=(2,2))(x3)

    #x4 = residue_block(x3, filters= 16, filter_size=(3,3), strides=(1,1))
    #x4 = MaxPooling2D(pool_size=(2,2))(x4)

    #x5 = residue_block(x4, filters=16, filter_size=(3,3), strides=(1,1))
    ##x5 = MaxPooling2D(pool_size=(2,2))(x5)

    #x6 = residue_block(x5, filters=16, filter_size=(3,3), strides=(1,1))
    #x6 = MaxPooling2D(pool_size=(2,2))(x6)

    x6 = Flatten()(x1)
    x6 = Dense(units=512, activation='relu')(x6)
    x6 = Dense(units=256, activation='relu')(x6)
    x6 = Dropout(0.75)(x6)
    output = Dense(units=10, activation='softmax')(x6)

    M = Model(inputs=inputs, outputs=output)
    adam = keras.optimizers.SGD(lr=args.lr, momentum=0.0, nesterov=False)
    # sgd = optimizers.SGD(lr=args.lr, momentum=0.0, decay=args.lr/args.epochs,nesterov=False)
    M.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    M.summary()

    return M


def mnn(input_shape=(height, width, 3)):
    #model
    inputs = Input(shape=(height, width, 1))
    #x1 = dilate(filters=1, kernel_size=(3,3),strides=(1,1),operation='m')(inputs)
    x1 = adapt(filters=16, kernel_size=(3,3),strides=(1,1),operation='m')(inputs)
    x1 = Activation('relu')(x1)
    #x1 = MaxPooling2D(pool_size=(2,2))(x1)

    #x1 = Add()([inputs, x1])
    #x1 = activations.relu(x1)
    x2 = adapt(filters=16, kernel_size=(3,3), strides=(1,1), operation='m')(x1)
    x2 = Activation('relu')(x2)
    #x2 = MaxPooling2D(pool_size=(2,2))(x2)

    #x3 = adapt(filters=16, kernel_size=(3,3), strides=(1,1), operation='m')(x2)
    #x3 = Activation('relu')(x3)
    #x3 = MaxPooling2D(pool_size=(2,2))(x3)

    x4 = Add()([inputs, x2])
    x5 = MaxPooling2D(pool_size=(2,2))(x4)

    #x4 = adapt(filters=64, kernel_size=(3,3), strides=(1,1), operation='m')(x3)
    #x4 = Activation('relu')(x4)
    #x4 = MaxPooling2D(pool_size=(2,2))(x4)

    #xx = Concatenate(axis=-1)([inputs, inputs])
    #for i in range(14):
    #    xx = Concatenate(axis=-1)([xx, inputs])

    #x3 = keras.layers.Subtract()([inputs, x2])
    #x3 = Activation('relu')(x3)

    x4 = Flatten()(x5)
    x5 = Dense(units=56, activation='relu')(x4)
    x6 = Dense(units=32, activation='relu')(x5)
    x6 = Dropout(0.75)(x6)
    output = Dense(units=10, activation='softmax')(x6)

    #x2 = Add()([x1, x2])
    #x2 = activations.tanh(x2)
    #x1 = keras.layers.Conv2D(filters=1,kernel_size=(3,3), strides=(1,1), padding='SAME')(inputs)
    #x2 = keras.layers.Conv2D(filters=1,kernel_size=(3,3), strides=(1,1), padding='SAME')(x1)
    #x3 = keras.layers.Conv2D(filters=1,kernel_size=(3,3), strides=(1,1), padding='SAME')(x2)
    #x2 = keras.activations.softmax(x1, axis=-1)
    #x2 = dilate(filters=1, kernel_size=(3,3), strides=(1,1),operation='m')(x1)
    #x3 = dilate(filters=1, kernel_size=(3,3),strides=(1,1),operation='m')(x2)
    #x4 = erode(filters=1, kernel_size=(3,3),strides=(1,1),operation='m')(x3)
    M = Model(inputs=inputs,outputs=output)
    adam = keras.optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #sgd = optimizers.SGD(lr=args.lr, momentum=0.0, decay=args.lr/args.epochs,nesterov=False)
    M.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    M.summary()

    return M

def train():
    cur_path = os.getcwd()
    checkpoint_path = cur_path + "/checkpoints/"
    if not os.path.exists(checkpoint_path): os.mkdir(checkpoint_path)
    model_checkpoint_save_path = os.path.join(checkpoint_path, args.model_name)
    if not os.path.exists(model_checkpoint_save_path): os.mkdir(model_checkpoint_save_path)
    #MNIST
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    ##Cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #x_train = x_train.reshape(x_train.shape[0], 28, 28, 3)
    #x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    print('x_train shape:', x_train.shape)
    print('x_test shape', x_test.shape)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') /255.

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)


    #y_train, kernel = get_y(x_train)
    #print(kernel)
    #y_test = get_y(x_test)

    M = res_mnn()
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(1, args.epochs+1):
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = 'Need: {:02d}:{:02d}:{:02d}'.format(need_hour, need_mins, need_secs)
        print('Epoch %d begins, ' % epoch, need_time)
        iterations = int(x_train.shape[0]/args.batch_size)
        #print(iterations)
        for step in range(0, iterations):
            idx = np.random.randint(0, x_train.shape[0], args.batch_size)
            train_imgs = x_train[idx]
            train_labels = y_train[idx]
            [loss, acc] = M.train_on_batch(train_imgs, train_labels)

            if step % 100 ==0 or step == iterations-1:
                val_idx = np.random.randint(0, x_test.shape[0], args.batch_size)
                val_imgs = x_test[val_idx]
                val_labels = y_test[val_idx]
                [val_loss, val_acc] = M.test_on_batch(val_imgs, val_labels)
                print('Epoch: %d, Step: %d, Train loss: %.4f, Train acc.: %.2f%%, Val loss: %.4f, Val acc.: %.2f%%' % (epoch, step, loss, 100*acc, val_loss, 100*val_acc))

        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        if epoch % 10 == 0 or epoch == args.epochs or epoch == 1:
            checkpoint_name = os.path.join(checkpoint_path, args.model_name + '/model_epoch_' + str(epoch) +'h5')
            M.save(checkpoint_name)

    #history = M.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,verbose=1)
    score = M.evaluate(x_test, y_test, verbose=0)
    M.save(args.model_name)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    weights = M.get_weights()
    #print(kernel)
    print("whoops!")
    predict1 = np.reshape(M.get_weights()[0], (3,3))
    predict2 = np.reshape(M.get_weights()[3], (3,3))
    print('adapt weight 1:', indicator(weights[2]))
    print('adapt weight 2:', indicator(weights[5]))
    print('adapt weight 3:', indicator(weights[8]))
    #print(np.sum(abs(predict-kernel1)))
    print('predict weight 1:', predict1)
    print('predict weight 2:', predict2)


if __name__ =='__main__':
    print('................')
    train()



#sum_dis = 0
#for i in range(100):
#    M.fit(x_train, y_train, batch_size = 64, epochs=100, verbose=1)
#    predict = np.reshape(np.round(M.get_weights()[0]), (5,5)).astype(int)
#    if np.array_equal(predict, kernel1):
#        sum_dis += 1
#
#print(sum_dis)

#similarity = sum_dis/100
#print(sum_dis)
#print(similarity)

#M.fit(x_train, y_train, batch_size = 64, epochs=20, verbose=1)
#predict = np.reshape(M.get_weights()[0], (3,3))
#print('predict SE:' ,predict)
#print('target SE:' , kernel1)
#if np.array_equal(predict, kernel1):
#    print('yes!')
#print('bias:' ,M.get_weights()[1])
#np.save('bias.npy', M.get_weights()[1])
#origin = x_train[1]
#target = y_train[1]
#prediction = M.predict(np.reshape(origin, [1,28,28,1]))
#print('input image:', np.reshape(origin, [28,28]))
#print('target image:' ,np.reshape(origin, [28,28]))
#print('predicted image:', np.reshape(origin, [28,28]))
#np.save('input_image.npy', np.reshape(origin, [28, 28]))
#np.save('target_image.npy', np.reshape(target, [28, 28]))
#np.save('predicted_image.npy', np.reshape(prediction, [28, 28]))



#samples = 4

#origin = io.imread('Lena.jpg', as_grey=True)
#origin = resize(origin, (28, 28))
#kernel = np.ones((3,3),np.uint8)
#kernel[0,0]=1
#kernel[1,1]=1
#kernel[2,2]=1
#print(kernel)
#target = cv2.dilate(origin,kernel,iterations=1)
#print(origin.shape)

#numbers = np.random.randint(low=0, high=5000, size=samples)
#plt.figure()
#for i in range(samples):
#    index = i + 1
#
#    origin = x_train[numbers[i]]
#    target = y_train[numbers[i]]
#
#    plt.subplot(3, samples, index + samples * 0)
#    image = np.reshape(origin, [28, 28])
#    plt.imshow(image, cmap='gray')
#    plt.axis('off')
#    #plt.title('original')
#
#    plt.subplot(3, samples, index + samples * 1)
#    image = np.reshape(target, [28, 28])
#    plt.imshow(image, cmap='gray')
#    plt.axis('off')
#    #plt.title('target')
#
#    plt.subplot(3, samples, index + samples * 2)
#    prediction = M.predict(np.reshape(origin, [1,28, 28,1]))
#    image = np.reshape(prediction, [28, 28])
#    plt.imshow(image, cmap='gray')
#    plt.axis('off')
#    #plt.title('prediction')
#
#plt.tight_layout()
#plt.show()
#plt.savefig('dilation_gray.png')
