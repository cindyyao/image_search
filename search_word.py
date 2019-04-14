from keras.engine.saving import load_model
from argparse import ArgumentParser

import utils 

def build_parser():
    par = ArgumentParser()
    par.add_argument('--input_word', type=str,
                     dest='input_word', help='input word to search query', required=True)
    par.add_argument('--features_path', type=str,
                     dest='features_path', help='filepath to save/load features', default='feature_word')
    par.add_argument('--file_mapping', type=str,
                     dest='file_mapping', help='filepath to save/load file to image mapping', default='index_word')
    par.add_argument('--glove_path', type=str,
                     dest='glove_path', help='path to pre-trained GloVe vectors', default='models/glove.6B')
    return par

def build_search(images_features, file_index):

    word_vectors = utils.load_glove_vectors(glove_path)
    image_index = utils.index_features(images_features, dims=300)
    results = utils.search_index_by_value(word_vectors[input_word], image_index, file_index)
    print(results)

if __name__ == "__main__":
    parser = build_parser()
    options = parser.parse_args()
    features_path = options.features_path
    file_mapping = options.file_mapping
    input_word = options.input_word
    glove_path = options.glove_path

    images_features, file_index = utils.load_features(features_path, file_mapping)
    build_search(images_features, file_index)
