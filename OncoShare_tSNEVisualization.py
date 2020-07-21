from datetime import datetime
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Plot t-SNE visualization of word embedding clusters 
def display_closestwords_tsnescatterplot(words):
    # Load the word embeddings
    model = KeyedVectors.load_word2vec_format("oncoshare_w2v.bin", binary=True)

    arr = np.empty((0,300), dtype='f') # embeddings of all visualized words
    word_labels = [] # all words being visualized

    for word in words: # for all seed words
        # add to labels vector and arr 
        word_labels.append(word) 
        arr = np.append(arr, np.array([model[word]]), axis=0)

        # get close words
        close_words = model.similar_by_word(word)

        # add the vector for each of the closest words to the array
        for wrd_score in close_words:
            wrd_vector = model[wrd_score[0]]
            word_labels.append(wrd_score[0])
            arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    # display each point on the scatter plot
    colors = ["darkcyan", "forestgreen", "mediumvioletred", "coral", "sienna"] # colors for each cluster
    for i in range(len(words)):
        plt.scatter(x_coords[i*11:(i+1)*11], y_coords[i*11:(i+1)*11], color = colors[i], s = 50, label = words[i] + "-related terms")

    # label each point with the word
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, color="black", xy=(x, y), xytext=(2, 2), size = 14, textcoords='offset points')
    plt.legend()

    # create the plot
    plt.xlim(x_coords.min()-10, x_coords.max()+10)
    plt.ylim(y_coords.min()-10, y_coords.max()+10)
    plt.show()

display_closestwords_tsnescatterplot(["mastectomy", "stage", "biopsy", "gene", "smoke"])



# Plot t-SNE visualization of document vectors (numNotes = number of notes for each class)
def display_documents_tsnescatterplot(numNotes):
    vec = np.load("data/OncoShare/xDataset.npy") # load doc vectors
    processedNotes = pd.read_csv("data/OncoShare/ProcessedNotes_Labeled.csv", lineterminator='\n') # load dataframe with labels

    # getting the indices of numNotes notes for each class
    zeroIndices = [] # indices with no recurrence label
    oneIndices = [] # indices with recurrence label
    noteIndex = 0
    while len(oneIndices) != numNotes:
        if processedNotes["LABEL"][noteIndex] == 0 and len(zeroIndices) < numNotes:
            zeroIndices.append(noteIndex)
        elif processedNotes["LABEL"][noteIndex] == 1:
            oneIndices.append(noteIndex)
        noteIndex += 1

    # concatenate the vectors into a numpy array
    zeroVec = np.array(vec[zeroIndices])
    oneVec = np.array(vec[oneIndices])
    vecs = np.concatenate((zeroVec, oneVec))

    # tsne dim reduction
    tsne = TSNE(n_components=2)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vecs)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    #create the plot
    plt.rcParams.update({'font.size': 16})
    plt.scatter(x_coords[0:numNotes], y_coords[0:numNotes], color = "blue", label = "No recurrence (12 months)")
    plt.scatter(x_coords[numNotes:numNotes*2], y_coords[numNotes:numNotes*2], color = "red", label = "Recurrence (12 months)")
    plt.legend()
    plt.show()


display_documents_tsnescatterplot(500)
