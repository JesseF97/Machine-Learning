# Read data
import pandas as pd
df = pd.read_csv(r"/Users/jessefrederick/Desktop/Machine_Learning.csv")

# Defining x and y
x = df.iloc[:,3:5].values
y = df.iloc[:,-1].values

###### Preprocessing

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 100)

#train test validation split
from sklearn.model_selection import train_test_split
x_main, x_test, y_main, y_test = train_test_split(x, y, test_size = 50)
y_train, x_val, x_train, y_val = train_test_split(x_main, y_main, test_size = 50)

###### Scaling
# Some models get confused of size of data in different columns

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)

# StandardScaler
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#x_train = scaler.fit_transform(x_train)
#x_test = scaler.transform(x_test)
#
#print(x_train)

###### K Nearest Neighbors

# Fitting model
#from sklearn.neighbors import KNeighborsClassifier
#model = KNeighborsClassifier(n_neighbors = 7)
#model.fit(x_train, y_train)

# Prediction
#y_pred = model.predict(x_test)

# Evaluation
#from sklearn.metrics import accuracy_score
#acc = accuracy_score(y_test, y_pred)
#print(f"accuracy: {acc}")

# Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred, labels = [0, 1])
#print("Confusion Matrix:")
#print(cm)

# Normalized Confusion Matrix
#import numpy as np
#print("Normalized Confusion Matrix:")
#print(cm / np.sum(cm, axis = 1).reshape(-1,1))
#
## Heatmap
#import seaborn as sns
#import matplotlib.puplot as plt
#sns.heatmap(cm, cmap = "Greens", annot = True,
#            cbar_kws = {"orientation":"vertical", "label":"color bar"},
#            xticklabels = [0,1], yticklabels = [0,1])
#plt.xlabel("Predicted")
#plt.ylabel("Actual")
#plt.title(" Normalized Confusion Matrix")
#plt.show()
