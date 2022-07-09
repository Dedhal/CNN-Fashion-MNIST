import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from random import shuffle
from skimage.util.shape import view_as_blocks
from skimage import io, transform
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.layers import LeakyReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import warnings
warnings.filterwarnings('ignore')

# Hyperparameters

img_rows = 25
img_cols = 25

EPOCHS = 100
BATCH_size = 10
BATCH_size_64 = 64
VERBOSE = 1
NB_CLASSES = 13
VALIDATION_SPLIT = 0.2

# Data import

train_size = 100000
test_size = 20000

train = glob.glob("../datasets/chess-dataset/train_big/*.jpeg")
test = glob.glob("../datasets/chess-dataset/test_big/*.jpeg")

shuffle(train)
shuffle(test)

train = train[:train_size]
test = test[:test_size]

piece_symbols = ' pPnNbBrRqQkK'

def fen_from_filename(filename):
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]

#f, axarr = plt.subplots(1,5, figsize=(20, 30))

#for i in range(0, 5):
#    axarr[i].set_title(fen_from_filename(train[i]), fontsize=10, pad=20)
#    axarr[i].imshow(mpimg.imread(train[i]))
#    axarr[i].axis('off')

# NORNAL

def images(images_path, image_height, image_width):
    imges_list = []

    for image in tqdm(os.listdir(images_path)):
        path = os.path.join(images_path, image)

        path = cv2.imread(path)
        image = cv2.resize(image, (image_height, image_width))
        imges_list.append([np.array(image)])
    shuffle(imges_list)

    # Convert List into Array
    array_image = np.array(imges_list)

    # Removed Dimention
    images = array_image[:,0,:,:]
    return images

def onehot_from_fen(fen):
    eye = np.eye(13)
    output = np.empty((0, 13))
    fen = re.sub('[-]', '', fen)

    for char in fen:
        if char in '12345678':
            output = np.append(output, np.tile(eye[0], (int(char), 1)), axis = 0)
        else:
            idx = piece_symbols.index(char)
            output = np.append(output, eye[idx].reshape((1, 13)), axis = 0)

    return output

def fen_from_onehot(onehot):
    output = ''
    for j in range(8):
        for i in range(8):
            output += piece_symbols[onehot[j][i]]
        if j != 7:
            output += '-'

    for i in range(8, 0, -1):
        output = output.replace(' ' * i, str(i))

    return output

def process_image(img):
    downsample_size = 200
    square_size = int(downsample_size/8)
    img_read = io.imread(img)
    img_read = transform.resize(img_read, (downsample_size, downsample_size), mode='constant')
    tiles = view_as_blocks(img_read, block_shape=(square_size, square_size, 3))
    tiles = tiles.squeeze(axis=2)
    return tiles.reshape(64, square_size, square_size, 3)

def train_gen(features, labels, batch_size):
    for i, img in enumerate(features):
        y = onehot_from_fen(fen_from_filename(img))
        x = process_image(img)
        yield x, y

def pred_gen(features, batch_size):
    for i, img in enumerate(features):
        yield process_image(img)


# Model
model = Sequential()

#model.add(Convolution2D(15, kernel_size=2, activation='relu', input_shape=(img_rows, img_cols, 3)))
#model.add(MaxPool2D(2))
#model.add(Convolution2D(5, kernel_size=2, activation='relu'))
#model.add(Flatten())
#model.add(Dropout(0.3))
#model.add(Dense(13, activation='softmax'))

model.add(Conv2D(15, kernel_size=(3,3), activation='linear', input_shape=(img_rows,img_cols,3), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3,3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.3))
model.add(Dense(13, activation='softmax'))

# Compile the Model

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.summary()

# Fit Parameters

print("\nTraining Progress:\n------------------------")

train_model = model.fit(train_gen(train, None, 64), batch_size=BATCH_size_64, steps_per_epoch=train_size/EPOCHS, epochs=EPOCHS, verbose=VERBOSE, validation_steps=test_size//(EPOCHS+6), validation_freq=1, validation_data=(train_gen(test, None, 64)))

# Evaluate the model on Test Data

#res = (model.predict_generator(pred_gen(test, 64), steps = test_size).argmax(axis=1).reshape(-1, 8, 8))

#pred_fens = np.array([fen_from_onehot(onehot) for onehot in res])
#test_fens = np.array([fen_from_filename(fn) for fn in test])

#final_accuracy = (pred_fens == test_fens).astype(float).mean()

#print("Final Accuracy: {:1.5f}%".format(final_accuracy*100))

def display_with_predicted_fen(image):
    pred = model.predict(process_image(image)).argmax(axis=1).reshape(-1, 8, 8)
    fen = fen_from_onehot(pred[0])
    imgplot = plt.imshow(mpimg.imread(image))
    plt.axis('off')
    plt.title(fen)
    plt.show()

#display_with_predicted_fen(test[0])
#display_with_predicted_fen(test[1])
#display_with_predicted_fen(test[2])

model.save("chess_model_TD100000_VD20000_Basic-CNN.h5py")

accuracy = train_model.history['accuracy']
val_accuracy = train_model.history['val_accuracy']
loss = train_model.history['loss']
val_loss = train_model.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()