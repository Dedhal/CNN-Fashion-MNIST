import numpy as np

import matplotlib.image as mpimg
import keras
import glob
from skimage import io, transform
from skimage.util.shape import view_as_blocks

test = 'test_real.jpeg'

piece_symbols = 'prbnkqPRBNKQ'

def images(images_path, image_height, image_width):
    imges_list = []

    for image in tqdm(os.listdir(images_path)):
        path = os.path.join(images_path, image)

        
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
            output = np.append(output, np.tile(eye[12], (int(char), 1)), axis = 0)
        else:
            idx = piece_symbols.index(char)
            output = np.append(output, eye[idx].reshape((1, 13)), axis = 0)

    return output

def fen_from_onehot(onehot):
    output = ''
    for j in range(8):
        for i in range(8):
            if onehot[j][i] == 12:
                output += ' '
            else:
                output += piece_symbols[onehot[j][i]]
        if j != 7:
            output += '-'

    for i in range(8, 0, -1):
        output = output.replace(' ' * i, str(i))

    return output

def fen_from_filename(filename):
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]

def process_image(img):
    downsample_size = 200
    square_size = int(downsample_size/8)
    img_read = io.imread(img)
    img_read = transform.resize(img_read, (downsample_size, downsample_size), mode='constant')
    img_read = np.array(img_read)
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

print("Loading model...")
model = keras.models.load_model('chess_model_TD100000_VD20000_Basic-CNN.h5py')
print("Prediction")

#path = cv2.imread("test_real.png")


res = model.predict(process_image(test)).argmax(axis=1).reshape(-1, 8, 8)

print(fen_from_onehot(res[0]))
