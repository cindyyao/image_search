from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from argparse import ArgumentParser

import utils

def build_parser():
    par = ArgumentParser()
    par.add_argument('--model_save_path', type=str,
                     dest='model_save_path', help='filepath to save trained model', default='my_model.hdf5')
    par.add_argument('--checkpoint_path', type=str,
                     dest='checkpoint_path', help='filepath to save training checkpoints', default='checkpoint.hdf5')
    par.add_argument('--glove_path', type=str,
                     dest='glove_path', help='path to pre-trained GloVe vectors', default='models/glove.6B')
    par.add_argument('--dataset_path', type=str,
                     dest='dataset_path', help='path to dataset', default='dataset')
    par.add_argument('--num_epochs', type=int,
                     dest='num_epochs', help='number of epochs to train on', default=50)

    return par


if __name__ == "__main__":
    parser = build_parser()
    options = parser.parse_args()
    model_save_path = options.model_save_path
    checkpoint_path = options.checkpoint_path
    glove_path = options.glove_path
    dataset_path = options.dataset_path
    num_epochs = options.num_epochs

    # load embedding into a word:vec dictionary
    word_vectors = utils.load_glove_vectors(glove_path)

    # vectors: embedding of img's class name (folder name)
    images, vectors, image_paths = utils.load_paired_img_wrd(dataset_path, word_vectors)
    
    x, y = shuffle(images, vectors, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
    custom_model = utils.setup_custom_vgg()
    custom_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     epochs=num_epochs, batch_size=32, callbacks=[checkpointer])
    custom_model.save(model_save_path)
