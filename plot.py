import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from donut import complete_timestamp, standardize_kpi, Donut, DonutTrainer, DonutPredictor
from tensorflow import keras as K
from tfsnippet.modules import Sequential
from tfsnippet.utils import get_variables_as_dict, VariableSaver
import tensorflow.compat.v1 as tf
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
tf.disable_v2_behavior()

# path to the dataset
file_csv = "sample_data/cpu4.csv"

# Read the raw data.
data = pd.read_csv(file_csv)
timestamp = data["timestamp"]
values = data["value"]
labels = data["label"]
dataset_name = file_csv.split('.')[0]
print("0 Timestamps: {}".format(timestamp.shape[0]))
print("0 Labeled anomalies: {}".format(np.sum(labels == 1)))

# Complete the timestamp filling missing points with zeros, and obtain the missing point indicators.
timestamp, missing, (values, labels) = complete_timestamp(timestamp, (values, labels))
print("1 Timestamp:{}".format(len(timestamp)))
print("1 Labeled anomalies: {}".format(np.sum(labels == 1)))
print("1 Missing points: {}".format(np.sum(missing == 1)))

# Split the training and testing data.
test_portion = 0.1
test_n = int(len(values) * test_portion)
train_timestamp, test_timestamp = timestamp[:-test_n], timestamp[-test_n:]
train_values, test_values = values[:-test_n], values[-test_n:]
train_labels, test_labels = labels[:-test_n], labels[-test_n:]
train_missing, test_missing = missing[:-test_n], missing[-test_n:]
print("Rows in test set: {}".format(test_values.shape[0]))
print("Anomalies in test set: {}".format(np.sum(test_labels == 1)))


anomaly_ts = []
anomaly_v = []
for i in range(0,len(test_timestamp)):
    if test_labels[i] == 1:
        anomaly_ts.append(test_timestamp[i])
        anomaly_v.append(test_values[i])

print("anomaly points {}".format(len(anomaly_ts)))

plt.plot(test_timestamp,test_values)
plt.scatter(anomaly_ts,anomaly_v,color=(1,0,0))
plt.show()