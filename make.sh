mkdir data/interim
mkdir data/processed

ipython src/data/convert_to_wav.py
ipython src/data/make_noise.py
ipython src/data/make_make_dataset_5.py


#ipython src/models/BLSTM_A5.py
# epoch 27 was initially chosen
