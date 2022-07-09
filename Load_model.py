import numpy as np

import matplotlib.image as mpimg
import keras
import glob
from skimage import io, transform
from skimage.util.shape import view_as_blocks

test = 'test_real.jpeg'

piece_symbols = ' pPnNbBrRqQkK'

def onehot_from_fen(fen):
    eye = np.eye(13)
    output = np.empty((0, 13))
    fen = re.sub('[-]', '', fen)

    for char in fen:
        if char in '12345678':
            output = np.append(output, np.tile(eye[12], (int(char), 1)), axis = 0)
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
    img_read = np.array(img_read)
    tiles = view_as_blocks(img_read, block_shape=(square_size, square_size, 3))
    tiles = tiles.squeeze(axis=2)
    return tiles.reshape(64, square_size, square_size, 3)

print("Loading model...")
model = keras.models.load_model('chess_model_TD100000_VD20000_Basic-CNN.h5py')
print("Prediction")

#path = cv2.imread("test_real.png")


res = model.predict(process_image(test)).argmax(axis=1).reshape(-1, 8, 8)

print(res[0])
print(fen_from_onehot(res[0]))
