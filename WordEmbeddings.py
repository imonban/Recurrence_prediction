import pandas as pd
from Preprocessing import process
import numpy as np
import os
from time import time
from gensim.models import Word2Vec, KeyedVectors

### Load all notes and process them

# MIMIC
MIMICnotes = pd.read_csv("data/MIMIC_data/NOTEEVENTS.csv")
processedNotes = MIMICnotes['TEXT'].apply(process)
del MIMICnotes

# I2B2
for i in os.listdir("data/I2B2"): #for all files under the I2B2 folder
    if (i.endswith(".txt")): #if they are .txt files
        f = open("data/I2B2/" + i, "r")
        note = f.read()
        processedNotes.append(process(note))

# OncoShare
oncoshareNotes = pd.read_csv("data/OncoShare/STANFORD_NOTE_DATA_TABLE.csv", encoding="ISO-8859-1")
processedNotes.extend(oncoshareNotes['NOTE'].apply(process))
del oncoshareNotes

processedNotes.apply(lambda x: x.split()) # split the words for word2vec training

# Train word2vec
t = time()
print('BUILDING MODEL')
new_wv = Word2Vec(processedNotes, size=300, window=30, min_count=50, sg=1, compute_loss=True, iter = 30)
print('Time to build the model (30 epochs): {} mins'.format(round((time() - t) / 60, 2)))
new_wv.wv.save_word2vec_format('oncoshare_w2v.bin', binary=True)


# Simple test of word embeddings
preTrainedPath = "oncoshare_w2v.bin"
t = time()
onco_wv = KeyedVectors.load_word2vec_format(preTrainedPath, binary=True)
print('Time to read the model: {} mins'.format(round((time() - t) / 60, 2)))
print('----------------------------')
print(onco_wv.most_similar(positive=['world']))
print(onco_wv.most_similar(positive=['p53']))
print(onco_wv.most_similar(positive=['mitosis']))
print(onco_wv.most_similar(positive=['triple']))
print('----------------------------')

