import numpy as np
import pandas as pd
from datetime import datetime
from numpy.random import seed
seed(1)
import matplotlib.pyplot as plt

#store the processed notes array
processedNotes = pd.read_csv("data/OncoShare/WeakLabelDataset_1.csv", encoding="ISO-8859-1", engine='python')

#create array of sorted triples (patient id, date, index)
def sortNotes():
    sortedArray = []
    for i in range(len(processedNotes)):
        noteDate = str2Date(processedNotes["NOTE_DATE"][i]) #store note date
        diff = noteDate - datetime.strptime('01/01/80', '%m/%d/%y') #calculate the number of day since 1/1/80
        triple = (processedNotes["ANON_ID"][i], diff.days, i) #create triple (patient id, date, index)
        sortedArray.append(triple) #add to the array
    sortedArray.sort() #sort the array
    return sortedArray


# convert the note label to a categorical array [no recur, recur, padded]
def labelToCategorical(x):
    if x == 0: return [1, 0, 0] #no recur
    elif x == 1: return [0, 1, 0] #recur
    else: return [0, 0, 1] #padded


# convert string s to date
def str2Date(s):
    if("/" in s):
        return datetime.strptime(s, '%m/%d/%y')
    else:
        return datetime.strptime(s, '%d-%m-%y')


#Returns xDATA, yDATA, and visit dates in the format for model training (padVal = max # notes per patient)
def prepData(padVal):
    noteVectors = [] # 3d array of all note vectors [patient_id, note_id, vector_element]
    labels = [] # 3d array of all patient labels [patient_id, note_id, categorical_element]
    dates = [] # 2d array of all visit dates [patient_id, note_id]

    noteVecs = [] # 2d array of a patient's note vectors
    labs = [] # 2d array of a patient's labels
    dats = [] # 1d array of a patient's visit dates

    vec = np.load("data/OncoShare/xDataset.npy") #load the xDataset
    sorted = sortNotes() #get the sorted array of the dataset
    curPatID = sorted[0][0] #keeps track of the ID of the patient currently being looked at
    
    for j in range(len(sorted)): # go through the sorted list of notes
        if (curPatID == sorted[j][0] and len(noteVecs) != padVal): # if same patient and max notes not reached, add visit data
                noteVecs.append(vec[sorted[j][2]])
                labs.append(labelToCategorical(processedNotes["LABEL"][sorted[j][2]]))
                dats.append(str2Date(processedNotes["NOTE_DATE"][sorted[j][2]]))

        else: # if diff patient
            while(len(noteVecs) != padVal): #pad visit sequence
                noteVecs.append(np.zeros(300))
                labs.append(labelToCategorical(2))
                dats.append(dats[-1])
            # append all patient info to the larger arrays
            noteVectors.append(noteVecs)
            labels.append(labs)
            dates.append(dats)
            #reset patient data
            curPatID = sorted[j][0]
            noteVecs = []
            labs = []
            dats = []
            #append the 1st visit data
            noteVecs.append(vec[sorted[j][2]])
            labs.append(labelToCategorical(processedNotes["LABEL"][sorted[j][2]]))
            dats.append(str2Date(processedNotes["NOTE_DATE"][sorted[j][2]]))
    
    while(len(noteVecs) != padVal): #padding for the last patient
        noteVecs.append(np.zeros(300))
        labs.append(labelToCategorical(2))
        dats.append(dats[-1])

    #append the last patient
    noteVectors.append(noteVecs)
    labels.append(labs)
    dates.append(dats)

    return (noteVectors, labels, dates) #return the larger arrays


### Save Prepared Data
data = prepData(800)
np.save("data/OncoShare/xData", np.array(data[0])) #save the xData
np.save("data/OncoShare/yData", np.array(data[1])) #save the yData
np.save("data/OncoShare/noteDates", np.array(data[2])) #save the visit dates


### Distribution of Followup
noteDates = np.load("data/OncoShare/noteDates.npy")

followup = []
for i in range(len(noteDates)):
    timeDiff = noteDates[i][799] - noteDates[i][0]
    followup = timeDiff.years

plt.rcParams.update({'font.size': 16})
cm = plt.cm.get_cmap('RdYlBu_r')
Y,X = np.histogram(followup, 25)
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]
plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
#plt.yscale('Log')
plt.ylabel('Number of Patients')
plt.xlabel('Length of Followup (years)')
plt.title('Distribution of Visits Per Patient')
plt.show()


### Distribution of Number of Notes
xData = np.load("data/OncoShare/xData.npy")
yData = np.load("data/OncoShare/yData.npy")

noteCounts = []
for i in range(len(xData)):
    for j in range(len(xData[i])):
        if (yData[i][j][2] == 1):
            noteCounts.append(j)
            break

plt.rcParams.update({'font.size': 16})
cm = plt.cm.get_cmap('RdYlBu_r')
Y,X = np.histogram(noteCounts, 25)
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]
plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
plt.yscale('Log')
plt.ylabel('Number of Patients')
plt.xlabel('# of visits (log scaled)')
plt.title('Distribution of Visits Per Patient')
plt.show()

