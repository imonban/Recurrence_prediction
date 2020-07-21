import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, BatchNormalization, Dropout
from keras.optimizers import Adam
from numpy import random
random.seed(1)

### Load Data
xData = np.load("data/OncoShare/xData_Weak2.npy")
yData = np.load("data/OncoShare/yData_Weak2.npy")

### Arrays to store model performance metrics
prec = []
rec = []
spec = []
f1 = []
fprs = []
tprs = []
aucs = []

### k-fold CV
skf = KFold(n_splits=5)
skf.get_n_splits(xData, yData)
mean_fpr = np.linspace(0, 1, 100)
counter = 0 # index of the split
colors = ["blue", "green", "orange", "purple", "red"] # for the ROC Curve
for train_index, test_index in skf.split(xData, yData): # run through each fold
    trainX, testX = xData[train_index], xData[test_index]
    trainY, testY = yData[train_index], yData[test_index]

    # Model Architecture
    model = Sequential()
    model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(25, input_shape=(trainX.shape[2], trainX.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(3, activation='softmax')))
    model.compile(sample_weight_mode="temporal", loss='categorical_crossentropy', optimizer=Adam(lr=0.001, amsgrad=True, decay=0.00001), metrics=['accuracy'])
    print(model.summary())

    # Count the number of labels per class
    class_count = [0,0,0]
    training_y = trainY.reshape(trainY.shape[0]*trainY.shape[1],3)
    for i in range(training_y.shape[0]):
        for j in range(3):
            if training_y[i][j] == 1: class_count[j] += 1

    # Set Weights for each training sample based on inverse class frequency
    sample_weights = []
    class_weights = [training_y.shape[0]/class_count[0], training_y.shape[0]/class_count[1], training_y.shape[0]/class_count[2]]
    for i in range(trainY.shape[0]):
        weights = []
        for j in range(trainY.shape[1]):
            for k in range(3):
                if trainY[i][j][k] == 1: weights.append(class_weights[k])
        sample_weights.append(weights)
    sample_weights = np.array(sample_weights)

    # Train the model
    model.fit(trainX, trainY, epochs=20, batch_size=32, sample_weight=sample_weights)

    # Test on test set
    pred = model.predict_proba(testX)
    testY = testY.reshape(testY.shape[0]*testY.shape[1],3) # reshape array for easier comparison
    pred = pred.reshape(pred.shape[0]*pred.shape[1],3) # reshape array for easier comparison

    # Create array deleteIndex for all padded data
    deleteIndex = []
    for i in range(testY.shape[0]):
        if (testY[i][2] == 1):
            deleteIndex.append(i)
        else: pred[i][1] = pred[i][1] / (1 - pred[i][2])

    # Remove all padded data before computing results
    testY = np.delete(testY, deleteIndex, 0)
    pred = np.delete(pred, deleteIndex, 0)

    # Plot ROC Curve
    fpr, tpr, threshold = metrics.roc_curve(testY[:, 1], pred[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    counter += 1
    plt.plot(fpr, tpr, color=colors[counter-1], lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (counter, roc_auc))

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(roc_auc)

    # Find optimal threshold
    i = np.arange(len(tpr))
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]

    # Generate the binary predictions based on optimal threshold
    binPred = []
    for pred in pred[:, 1]:
        if (pred > optimal_threshold): binPred.append(1)
        else: binPred.append(0)
    binPred = np.array(binPred)

    # Calculate metrics
    prec.append(precision_score(testY[:, 1],binPred))
    rec.append(recall_score(testY[:, 1],binPred))
    tn, fp, fn, tp = confusion_matrix(testY[:, 1],binPred).ravel()
    spec.append(tn/(tn+fp))
    f1.append(f1_score(testY[:, 1],binPred))


### Print out metrics
print("ROC AUC")
roc = np.array(aucs)
print(np.mean(roc))
print(np.std(roc))
print()
print("PRECISION")
prec = np.array(prec)
print(np.mean(prec))
print(np.std(prec))
print()
print("RECALL/SENSITIVITY")
rec = np.array(rec)
print(np.mean(rec))
print(np.std(rec))
print()
print("SPECIFICITY")
spec = np.array(spec)
print(np.mean(spec))
print(np.std(spec))
print()
print("F1")
f1 = np.array(f1)
print(np.mean(f1))
print(np.std(f1))


### Plot the ROC Curve
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

plt.xlim((-0.05,1.05))
plt.ylim((-0.05, 1.05))
plt.legend(loc="lower right")
plt.show()
