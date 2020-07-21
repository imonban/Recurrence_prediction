import pandas as pd
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from Preprocessing import process
import numpy as np
from datetime import datetime

#Convert string to datetime object
def str2Date(s):
    if("/" in s):
        return datetime.strptime(s, '%m/%d/%y')
    else:
        return datetime.strptime(s, '%d-%m-%y')


### INITIAL COMPILATION OF NOTES WITH LABELS
y = pd.read_excel("data/OncoShare/Oncoshare_v3_fullannotation_prob.xlsx") #read labels (change file based on different labeling strategy)
notes = pd.read_csv("data/OncoShare/STANFORD_NOTE_DATA_TABLE.csv", encoding="ISO-8859-1") #read notes

notes_df = pd.DataFrame(columns = ["ANON_ID", "NOTE_DATE", "NOTE_TYPE", "NOTE", "PROCESSED"]) #create dataframe for all notes with labels
index = 0

labeled = [False for i in range(100000)] #boolean array to check if a patient has labels
for i in range(0,len(y)):
    labeled[int(y["ANON_ID"][i])] = True

for j in range(0,len(notes)):
    if labeled[notes["ANON_ID"][j]]: #check if the note has a label
        if "/" in notes["NOTE_DATE"][j] or "-" in notes["NOTE_DATE"][j]: #check if the note has an actual date
            processedNote = " ".join(process(notes["NOTE"][j])) # process the raw note
            if isinstance(processedNote, str) and len(processedNote) > 0: # check that the processed note is valid
                notes_df.loc[index] = [notes["ANON_ID"][j], notes["NOTE_DATE"][j], notes["NOTE_TYPE"][j], notes["NOTE"][j], processedNote] #add the id, date, type, raw note, and processed note to the dataframe
                index += 1
notes_df.to_csv("data/OncoShare/ProcessedNotes.csv", index=False) #save the dataframe


### DOCUMENT VECTORIZATION (TF-IDF * word2vec)
processedNotes = pd.read_csv("data/OncoShare/ProcessedNotes.csv", encoding="ISO-8859-1", engine='python') #read the processed notes dataframe
reports = processedNotes["PROCESSED"] #store the processed notes in reports

tfidf = TfidfVectorizer(sublinear_tf=True)
features = tfidf.fit_transform(processedNotes["PROCESSED"]) #tfidf on the processed notes
dict = {val : idx for idx, val in enumerate(tfidf.get_feature_names())} #create a dictionary mapping each word to their index in the tfidf array

wv = KeyedVectors.load_word2vec_format("oncoshare_w2v.bin", binary=True) #read pretrained word embeddings

docVecs = [] #array for document vectors
for i in range(len(reports)):
    report = reports[i].split() #split the processed note into list of each word
    avgFeat = np.zeros(300) #set each doc vector to 300 0's
    for word in report:
        try: #if in the vocab of the embeddings
            vector = np.multiply(wv[word],features[i,dict[word]]) #mutliply word embedding * tfidf of the word
            avgFeat = np.add(avgFeat, vector) #add the weighted word embedding to the doc vector
        except: # if OOV
            pass
    docVecs.append(avgFeat/len(report)) #append the average of all weighted word embeddings

np.save("data/OncoShare/xDataset", docVecs) #save the xDataset


# NOTE LABELING
y = pd.read_excel("data/OncoShare/Oncoshare_v3_fullannotation_prob.xlsx") #read labels
earliestRecur = np.array([datetime(5000, 1, 1) for i in range(100000)]) #create array for earliest recurrence data (set to 1/1/5000 for patients with no recurrence)
for i in range(len(y)):
    if ("Definite recurrence" in y["Predicted"][i]): #if the label indicates recurrence
        labelDate = datetime.strptime(y["START_DATE"][i], '%Y-%m-%d') #read the date of the label
        if (labelDate < earliestRecur[int(y["ANON_ID"][i])]): #check if label's date is earlier than the existing earliest recurrence
            earliestRecur[int(y["ANON_ID"][i])] = labelDate; #set to the new earliest occurence

processedNotes = pd.read_csv("data/OncoShare/ProcessedNotes.csv", encoding="ISO-8859-1", engine='python') #read the processed notes dataframe
processedNotes["LABEL"] = [0 for i in range(len(processedNotes))] #all labels initially 0

for noteIndex in range(len(processedNotes)):
    noteDate = str2Date(processedNotes["NOTE_DATE"][noteIndex]) #store the date of the note
    label = earliestRecur[int(processedNotes["ANON_ID"][noteIndex])] #store the date of earliest recurrence
    diff = label - noteDate #calculate diff
    if (diff.days < 365): #set to appropriate # of days
        processedNotes["LABEL"][noteIndex] = 1
processedNotes.to_csv("data/OncoShare/ProcessedNotes_Labeled.csv", index=False) #save the labels

