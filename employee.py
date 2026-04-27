Importing Libraries
# for numerical computing
import numpy as np
# for dataframes
import pandas as pd
# for easier visualization
import seaborn as sns
# for visualization and to display plots
from matplotlib import pyplot as plt
%matplotlib inline
# import color maps
from matplotlib.colors import ListedColormap
# to split train and test set
from sklearn.model_selection import train_test_split
# to perform hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
#import xgboost
import os
# To save the final model on disk
from sklearn.externals import joblib
np.set_printoptions(precision=2, suppress=True)
df = pd.read_csv("D:/Pandas/employee_data.csv")
# Dataframe dimensions
df.shape
# Columns of the dataframe
df.columns
df.head()
df.describe()

Data Visualization
## We can also use bar plots instead
plt.figure(figsize=(6,4))
sns.countplot(y='department', data=df)
# Plot histogram grid
df.hist(figsize=(6,9), xrot=-45)
## status vs satisfaction
sns.boxplot(y='status', x='satisfaction', data=df)
## status vs last_evaluation
sns.boxplot(y='status', x='last_evaluation', data=df)
Correlations
df.corr()
plt.figure(figsize=(5,6))
sns.heatmap(df.corr())
# Drop duplicates
df = df.drop_duplicates()
print(df.shape)
# Drop temporary workers
df = df[df.department != 'temp']
print(df.shape)
Fix structural errors
# Print unique values of 'filed_complaint'
print( df.filed_complaint.unique() )
# Print unique values of 'recently_promoted'
print( df.recently_promoted.unique() )
# Missing filed_complaint values should be 0
df['filed_complaint'] = df.filed_complaint.fillna(0)
# Missing recently_promoted values should be 0
df['recently_promoted'] = df.recently_promoted.fillna(0)
# Print unique values of 'filed_complaint'
print( df.filed_complaint.unique() )
# Print unique values of 'recently_promoted'
print( df.recently_promoted.unique() )
# Plot class distributions for 'department'
sns.countplot(y='department', data=df)
Outliers
# Display number of missing values by feature
df.isnull().sum()
#Fill missing values in department with 'Missing'
# Display number of missing values by feature

df.isnull().sum()
One hot encoding
# Create new dataframe with dummy features
df = pd.get_dummies(df, columns=['department', 'salary'])
# Display first 10 rows
df.head(10)
# Save analytical base table
df.to_csv('D:/Pandas/analytical_base_table.csv', index=None)
# Create separate object for target variable
y = df.status
# Create separate object for input features
X = df.drop('status', axis=1)
# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=1234,
                                                    stratify=df.status)
# Print number of observations in X_train, X_test, y_train, and y_test
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train = (X_train - train_mean) / train_std

## Check for mean and std dev.
X_train.describe()
## Note: We use train_mean and train_std_dev to standardize test data set
X_test = (X_test - train_mean) / train_std
## Check for mean and std dev. - not exactly 0 and 1
X_test.describe()
Logistic Regression Code:
X_test.fillna(X_train.mean(), inplace=True)
tuned_params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'penalty': ['l1', 'l2']}
model = GridSearchCV(LogisticRegression(), tuned_params, scoring = 'roc_auc', n_jobs=-1)
model.fit(X_train, y_train)
model.best_estimator_
## Predict Train set results
y_train_pred = model.predict(X_train)
## Predict Test set results
y_pred = model.predict(X_test)
# Get just the prediction for the positive class (1)
y_pred_proba = model.predict_proba(X_test)[:,1]
# Display first 10 predictions
y_pred_proba[:10]

i=113  ## Change the value of i to get the details of any point (56, 213, etc.)
print('For test point {}, actual class = {}, precited class = {}, predicted probability = {}'.
      format(i, y_test.iloc[i], y_pred[i], y_pred_proba[i]))
confusion_matrix(y_test, y_pred).T
# Plot the ROC curve
fig = plt.figure(figsize=(5,5))
plt.title('Receiver Operating Characteristic')
# Calculate AUC for Train set
print(roc_auc_score(y_train, y_train_pred))
## Building the model again with the best hyperparameters
model = LogisticRegression(C=1000, penalty = 'l1')
model.fit(X_train, y_train)
indices = np.argsort(-abs(model.coef_[0,:]))
print("The features in order of importance are:")
print(50*'-')
for feature in X.columns[indices]:
    print(feature)
# Calculate AUC for Test set
print(auc(fpr, tpr))

Random Forest Code:
tuned_params = {'n_estimators': [100, 200, 300, 400, 500], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
model = RandomizedSearchCV(RandomForestClassifier(), tuned_params, n_iter=15, scoring = 'roc_auc', n_jobs=-1)
model.fit(X_train, y_train)
model.best_estimator_
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred).T
# Calculate AUC for Train set
roc_auc_score(y_train, y_train_pred)
# Calculate AUC for Test set
print(auc(fpr, tpr))
## Building the model again with the best hyperparameters
model = RandomForestClassifier(n_estimators=200, min_samples_split=2, min_samples_leaf=1)
model.fit(X_train, y_train)
indices = np.argsort(-model.feature_importances_)
print("The features in order of importance are:")
print(50*'-')
for feature in X.columns[indices]:
    print(feature)

K-Nearest Neighbour Code:
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# creating odd list of K for KNN
neighbors = list(range(1,20,2))
# empty list that will hold cv scores
cv_scores = []
#  10-fold cross validation , 9 datapoints will be considered for training and 1 for cross validation (turn by turn) to determine value of k
# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('\nThe optimal number of neighbors is %d.' % optimal_k)
y_pred = classifier.predict(X_test)
y_train_pred = classifier.predict(X_train)
acc = accuracy_score(y_test, y_pred, normalize=True) * float(100)  ## get the accuracy on testing data
acc
cnf=confusion_matrix(y_test,y_pred).T
cnf

# Get just the prediction for the positive class (1)
y_pred_proba = classifier.predict_proba(X_test)[:,1]
# Display first 10 predictions
y_pred_proba[:10]
# Calculate AUC for Train
roc_auc_score(y_train, y_train_pred)
# Calculate AUC for Test
print(auc(fpr, tpr))
Decision Tree Code:
tuned_params = {'min_samples_split': [2, 3, 4, 5, 7], 'min_samples_leaf': [1, 2, 3, 4, 6], 'max_depth': [2, 3, 4, 5, 6, 7]}
model = RandomizedSearchCV(DecisionTreeClassifier(), tuned_params, n_iter=15, scoring = 'roc_auc', n_jobs=-1)
model.fit(X_train, y_train)
model.best_estimator_
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]
y_pred_proba[:10]
confusion_matrix(y_test, y_pred).T
# Calculate AUC for Train
roc_auc_score(y_train, y_train_pred)

print(auc(fpr, tpr))
Support Vector Machine Code:
from sklearn import svm
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X_train, y_train)
    grid_search.best_params_
    return grid_search.best_params_
svClassifier=SVC(kernel='rbf',probability=True)
svClassifier.fit(X_train,y_train)
## Building the model again with the best hyperparameters
model = SVC(C=10, gamma=1)
model.fit(X_train, y_train)

## Predict Train results
y_train_pred = model.predict(X_train)
## Predict Test results
y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred).T
# Calculate AUC for Train
roc_auc_score(y_train, y_train_pred)
print(auc(fpr, tpr))
