#### Salary Based on Gender

# Read data
import pandas as pd
df = pd.read_csv(r"/Users/jessefrederick/Desktop/Machine_Learning_2.csv")

# Defining x and y
x = df.iloc[:,3:5].values
y = df.iloc[:,-1].values

###### Preprocessing

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 100) # add random_state = 0 after test_size to keep consistent f1 results.


#train test validation split
from sklearn.model_selection import train_test_split
x_main, x_test, y_main, y_test = train_test_split(x, y, test_size = 50)
x_train, x_val, y_train, y_val = train_test_split(x_main, y_main, test_size = 50)

###### Scaling
# Some models get confused of size of data in different columns

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)

# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)

###### K Nearest Neighbors

# Fitting model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 7)
model.fit(x_train, y_train)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression() #init random_state = 0 to keep consistent results. Make sure to add to test_size as well.
model.fit(x_train, y_train)

# Decision tree
from sklearn.tree import DecisionClassifier
model = DecisionClassifier() #init max_depth = int to specify the depth of the tree results; random_state = 0 for consistent results
model.fit (x_train, y_train)

# Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier() #add n_estimator to specifiy number of estimators; max_depth = int to specify the depth of the tree results; random_state = 0 for consistent results
model.fit(x_train, y_train)

## Linear Regression
from sklearn.model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
# LR Prediction
y_pred = model.predict(x_test)
#LR Evaluation
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print (f"R2: {r2}")
#LR Mean Squared Error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"RMSE: {np.sqrt(mse)}")

# Prediction
y_pred = model.predict(x_test)

# Evaluation
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(f"accuracy: {acc}")

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels = [0, 1])
print("Confusion Matrix:")
print(cm)

# Normalized Confusion Matrix
import numpy as np
print("Normalized Confusion Matrix:")
print(cm / np.sum(cm, axis = 1).reshape(-1,1))

## Heatmap
import seaborn as sns
import matplotlib.puplot as plt
sns.heatmap(cm, cmap = "Greens", annot = True,
            cbar_kws = {"orientation":"vertical", "label":"color bar"},
            xticklabels = [0,1], yticklabels = [0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Confusion Matrix")
plt.show()

# Recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall}")

# Precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision}")

# Specificity
specificity = cm[0,0]/(cm[0,0] + cm[0,1])
print(f"Specificity: {specificity}")

# f1 Score
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print(f"f1: {f1}")




