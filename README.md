# Semantic Image Search
A semantic image search engine that search images from image or text.
## Setup
Clone the repository locally and create a virtual environment. Check the `requirements.txt` for dependencies.
```
conda create --name semantic-search python=3.6
conda activate semantic-search
pip install -r requirements.txt
```
Run the ```Setup.sh``` to download pre-trained GloVe vectors and Pascal Sentence Dataset to train the CNN model.
```
sh setup.sh
```
## Model Training
Use pre-trained VGG model with output layer replaced by GloVe embedding of the class label, and re-train the model.
```
python train.py \
  --model_save_path my_model.hdf5 \
  --checkpoint_path checkpoint.hdf5 \
  --glove_path models/glove.6B \
  --dataset_path dataset \
  --num_epochs 50
  ```
  ## Indexing
  Index the image embeddings to `.npy` files.
  ```
  python index.py \
    --index_folder dataset \
    --word_features_path feature_word \
    --img_features_path feature_img \
    --word_file_mapping index_word \
    --img_file_mapping index_img \
    --model_path my_model.hdf5 \
    --glove_path models/glove.6B
  ```
  ## Searching
  Search for an image using image
  ```
  python search_image.py \
    --input_image dataset/cat/2008_001335.jpg \
    --features_path feature_img \
    --file_mapping index_img 
  ```
  Search for an image using text
  ```
  python search_word.py \
    --input_word cat \
    --features_path feature_word \
    --file_mapping index_word \
    --glove_path models/glove.6B
  ```
