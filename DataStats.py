import pandas as pd
import numpy as np
import os
from nltk.tokenize import sent_tokenize, word_tokenize

# I2B2 Document-Level Stats
wordStats = []
sentStats = []

for i in os.listdir("data/I2B2"): #for all files under the I2B2 folder
    if (i.endswith(".txt")): #if they are .txt files
        f = open("data/I2B2/" + i, "r")
        f = f.read()
        wordStats.append(len(word_tokenize(f))) #append # words
        sentStats.append(len(sent_tokenize(f))) #append # sentences

print(np.mean(wordStats))
print(np.std(wordStats))
print()
print(np.mean(sentStats))
print(np.std(sentStats))


# Compare Number of Positive Labels between 2 Weakly Supervised Approaches
numPositiveLabels = 0
processedNotes = pd.read_csv("data/OncoShare/WeakLabelDataset_1.csv", encoding="ISO-8859-1", engine='python') #1st Weak Supervision Strat
for noteIndex in range(len(processedNotes)):
    if processedNotes["LABEL"][noteIndex] == 1: #if the note is labeled positive
        numPositiveLabels += 1 #add to counter
print(numPositiveLabels)

numPositiveLabels = 0
processedNotes = pd.read_csv("data/OncoShare/WeakLabelDataset_2.csv", encoding="ISO-8859-1", engine='python') #2nd Weak Supervision Strat
for noteIndex in range(len(processedNotes)):
    if processedNotes["LABEL"][noteIndex] == 1: #if the note is labeled positive
        numPositiveLabels += 1 #add to counter
print(numPositiveLabels)

