# Make file structure
mkdir data/interim
mkdir data/processed
mkdir models
mkdir src/external

# Make softlinks
lre17_eval=/export/b16/janto/LRE17/corpora/lre17/data/eval/
lre17_dev=/export/corpora/LDC/LDC2017E23/LDC2017E23_2017_NIST_Language_Recognition_Evaluation_Development_Data/data/dev/
lre17_train=/export/corpora/LDC/LDC2017E22/LDC2017E22_2017_NIST_Language_Recognition_Evaluation_Training_Data/data/

ln -s ${lre17_eval} data/raw/lre17_eval
ln -s ${lre17_dev} data/raw/lre17_dev
ln -s ${lre17_train} data/raw/lre17_train

# Downloade and install external software
#PESQ
#STOI
#SDR


# Simulate data
ipython src/data/convert_to_wav.py  #uses 4 threads
ipython src/data/make_noise.py
ipython src/data/make_dataset_5.py

# How to train model


# How to enhance files in folder

#ipython src/models/BLSTM_A5.py
# epoch 27 was initially chosen
