from keras.engine.saving import load_model
from argparse import ArgumentParser

import utils 

def build_parser():
    par = ArgumentParser()
    par.add_argument('--features_path', type=str,
                     dest='features_path', help='filepath to save/load features', default='feature_img')
    par.add_argument('--file_mapping', type=str,
                     dest='file_mapping', help='filepath to save/load file to image mapping', default='index_img')
    par.add_argument('--input_image', type=str,
                     dest='input_image', help='input image path to search query', required=True)
    return par

def build_search(images_features, file_index, image_feature):
    image_index = utils.index_features(images_features, dims=4096)
    results = utils.search_index_by_value(image_feature, image_index, file_index)
    print(results)

if __name__ == "__main__":
    parser = build_parser()
    options = parser.parse_args()
    features_path = options.features_path
    file_mapping = options.file_mapping
    input_image = options.input_image

    model = utils.load_headless_pretrained_model()
    image = utils.load_img(input_image)
    image_feature = model.predict(image).reshape((4096,))
    print(image_feature.shape)
    images_features, file_index = utils.load_features(features_path, file_mapping)
    build_search(images_features, file_index, image_feature)

