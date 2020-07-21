import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, auc, roc_curve

# Visualize model predictions for specific patient (model predictions are averaged every x days to remove clutter)
def showGraph(patient, x):
    register_matplotlib_converters()

    # load predictions, labels, and note dates
    pred = np.load("data/OncoShare/yPred.npy")
    testY = np.load("data/OncoShare/yActual.npy")
    testDates = np.load("data/OncoShare/TestNoteDates.npy", allow_pickle=True)

    predValue = [] # array of model predictions
    realGraph = [] # array of ground truth labels
    dates = [] # array of visit dates
    noteIndex = 0 
    date = testDates[patient][noteIndex]
    preds = []
    real = 0
    while noteIndex < 800 and testY[patient][noteIndex][2] == 0: # while there are more unpadded notes
        if ((testDates[patient][noteIndex] - date).days < x): # to remove clutter, we look at x days at a time
            preds.append(pred[patient][noteIndex][1] / (1-pred[patient][noteIndex][2])) # append the predicted probability (scaled to remove padded class)
            if (testY[patient][noteIndex][1] == 1): real = 1 # check for recurrence
        else: # if we have looked at x days of notes

            #append the average prediction, label, date
            predValue.append(sum(preds)/len(preds))
            dates.append(date)
            realGraph.append(real)

            #reset the data to look at the next x days, starting with the current noteIndex
            date = testDates[patient][noteIndex]
            preds = [pred[patient][noteIndex][1] / (1-pred[patient][noteIndex][2])]
        noteIndex += 1

    # plot the predictions against true labels
    plt.figure()
    plt.ylim((-0.05, 1.05))
    plt.plot(dates, predValue, 'bo-', label='Predicted Recurrence (Probability)')
    plt.plot(dates, realGraph, 'ro-', label='Actual Recurrence (Binary)')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.xlabel("Date of Visit")
    plt.ylabel("Probability of Breast Cancer Recurrence (1 year)")
    plt.show()


# Examine Visualization of the first 10 patients in the test set
for patient in range(0,10):
    showGraph(patient)
