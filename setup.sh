pip install -r requirements.txt
curl -LO http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mkdir models
mkdir models/glove.6B
mv glove.6B.300d.txt models/glove.6B/
mkdir dataset
python prep_training.py
mv dataset/diningtable dataset/dining_table
mv dataset/pottedplant dataset/potted_plant
mv dataset/tvmonitor dataset/tv_monitor
