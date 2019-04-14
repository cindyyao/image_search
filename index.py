from keras.engine.saving import load_model
from argparse import ArgumentParser

import utils 

def build_parser():
    par = ArgumentParser()
    par.add_argument('--word_features_path', type=str,
                     dest='word_features_path', help='filepath to save/load word features', default='feature_word')
    par.add_argument('--img_features_path', type=str,
                     dest='img_features_path', help='filepath to save/load image features', default='feature_img')
    par.add_argument('--word_file_mapping', type=str,
                     dest='word_file_mapping', help='filepath to save/load file to word mapping', default='index_word')
    par.add_argument('--img_file_mapping', type=str,
                     dest='img_file_mapping', help='filepath to save/load file to image mapping', default='index_img')
    par.add_argument('--index_folder', type=str,
                     dest='index_folder', help='folder to index', default='dataset')
    par.add_argument('--glove_path', type=str,
                     dest='glove_path', help='path to pre-trained GloVe vectors', default='models/glove.6B')
    par.add_argument('--model_path', type=str,
                     dest='model_path', help='path to custom model', default='my_model.hdf5')
    return par

def generate_features(index_folder, features_path, file_mapping, loaded_model, glove_path):
    features, index = index_images(
        index_folder, 
        features_path, 
        file_mapping, 
        loaded_model,
        glove_path)
    print("Indexed %s images" % len(features))
    return features

def index_images(folder, features_path, mapping_path, model, glove_path):
    print ("Now indexing images...")
    word_vectors = utils.load_glove_vectors(glove_path)
    _, _, paths = utils.load_paired_img_wrd(
        folder=folder, 
        word_vectors=word_vectors)
    images_features, file_index = utils.generate_features(paths, model)
    utils.save_features(features_path, images_features, mapping_path, file_index)
    return images_features, file_index

# def build_feature_tree(file_name, features, n_trees=1000, dims=4096):
# 	feature_index = utils.index_features(features, n_trees, dims)
# 	utils.save_obj(file_name, feature_index)
# 	print('feature tree built!')

if __name__ == "__main__":
    parser = build_parser()
    options = parser.parse_args()
    word_features_path = options.word_features_path
    img_features_path = options.img_features_path
    word_file_mapping = options.word_file_mapping
    img_file_mapping = options.img_file_mapping
    index_folder = options.index_folder
    model_path = options.model_path
    glove_path = options.glove_path

    custom_model = load_model(model_path)
    features = generate_features(index_folder, word_features_path, word_file_mapping, custom_model, glove_path)

    vgg_model = utils.load_headless_pretrained_model()
    features = generate_features(index_folder, img_features_path, img_file_mapping, vgg_model, glove_path)


