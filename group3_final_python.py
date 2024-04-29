#!/usr/bin/env python
# coding: utf-8

# # Group 3 - Credit Card Fraud Detection
# ### Pre-processing  - Retrieve  & prepare the data:
# Loading and checking the data

# In[5]:




import pandas as pd
seed=42
df = pd.read_csv('D:/centennialcollege/winter2024/Comp247/groupproject/creditcard.csv')
df.head()


# In[12]:


print("Rows & Columns:\n", df.shape)
print("\nColumn Names:\n", list(df.columns))
print("\nColumn Names and data types\n", df.dtypes)


# In[13]:


print("0 = Normal Transaction, 1 = Fraud\n\n", df['Class'].value_counts())
print()
print(round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset are normal transactions')
print(round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset are fraud transactions')


# In[14]:


print("Missing Values in Dataset: ", df.isnull().sum().max())


# In[15]:


print("Stats:\n", df.describe())


# #### Data Visualization

# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Class', data=df)
plt.xlabel('Class')
plt.ylabel('Number of Transactions')
plt.title('Distribution of the amount of Normal Transactions (0) vs Fraud (1)')
plt.xticks([0, 1], ['Normal', 'Fraud'])
plt.show()


# In[17]:


sns.boxplot(x='Class', y='Amount', data=df)
plt.title('Distribution of Transaction Amounts by Class')
plt.ylim(0, 15000)
plt.xticks([0, 1], ['Normal', 'Fraud'])
plt.xlabel('Class')
plt.ylabel('Amount ($)')
plt.show()


# In[18]:


# Plotting the distribution of transaction time
plt.figure(figsize=(10, 6))
plt.hist(df['Time'], bins=50, alpha=0.7)
plt.xlabel('Transaction Time')
plt.ylabel('Frequency')
plt.title('Distribution of Transaction Time')
plt.show()


# In[19]:


#Correlation Heatmap of V1 to V28
corr_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".1f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# #### Data Cleaning & Transformation
# 
# Scale Time and Amount columns and create a sub sample of the dataframe in order to have an equal amount of Fraud and Non-Fraud cases

# In[20]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

pipeline = Pipeline([
    ('scaling', ColumnTransformer([
        ('scale_amount_time', StandardScaler(), [0, 29])  #Scaling 1st column(Time) and 30th column(Amount)
    ], remainder='passthrough'))
])

# Apply pipepline
df_transformed = pipeline.fit_transform(df)

# Extract scaled Time and Amount from transformed df
scaled_time = df_transformed[:, 0]
scaled_amount = df_transformed[:, 1]

# Drop the original 'Amount' and 'Time' columns from original df
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# Insert scaled 'Amount' and 'Time' back into df
df.insert(0, 'Scaled_Amount', scaled_amount)
df.insert(1, 'Scaled_Time', scaled_time)

df.head()


# Since Classes are highly skewed, have a normal distribution of the classes using Random User-Sampling.

# In[21]:


from imblearn.combine import SMOTEENN
import pandas as pd
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  


X = df.drop('Class', axis=1)
y = df['Class']


# Combination of oversampling and undersampling using SMOTEENN
smoteenn = SMOTEENN(random_state=seed)
X_resampled_smoteenn, y_resampled_smoteenn = smoteenn.fit_resample(X, y)


# In[22]:


from sklearn.utils import resample


# Sample size 
sample_size = 20000

# Randomly sample from the dataset
X_sampled_random, y_sampled_random = resample(X_resampled_smoteenn, y_resampled_smoteenn, n_samples=sample_size, random_state=seed)


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt


# Visualize the distribution of classes after using SMOTEENN
print('Distribution of the Classes in the SMOTEENN resampled dataset')
print(y_sampled_random.value_counts()/len(y_sampled_random))

sns.countplot(x=y_sampled_random)
plt.title('New Equally Distributed Classes after SMOTEENN', fontsize=14)
plt.show()


# # Split data into 70% training and 30% testing

# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

# Separate Features from class



# Split the data into training and testing sets
seed = 42
train_X, test_X, train_y, test_y = train_test_split(X_sampled_random , y_sampled_random, test_size=0.3, random_state=seed)

# Feature selection on the training data
selector = SelectKBest(f_classif, k=10)
train_X_new = selector.fit_transform(train_X, train_y)
best_features_indices = selector.get_support(indices=True)
print("Best Features: ", train_X.columns[best_features_indices])

# Transform testing data using the selected features
test_X_new = selector.transform(test_X)


# # Logistic Regression
# 

# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
#  Logistic regression model
linear_clf = LogisticRegression()
linear_clf.fit(train_X, train_y)

# Predictions and Evaluation
linear_predictions = linear_clf.predict(test_X)
print("Linear Classifier - Classification Report")
print(classification_report(test_y, linear_predictions))
print("Confusion Matrix")
print(confusion_matrix(test_y, linear_predictions))


# In[26]:


# Fine tune Logistic Regression model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'penalty': ['l1', 'l2'],  # Regularization penalty
    'solver': ['liblinear']  # Algorithm for optimization
}

logistic_regression = LogisticRegression()

# grid search with cross-validation
grid_search = GridSearchCV(estimator=logistic_regression, param_grid=param_grid, cv=5, scoring='f1')
grid_search.fit(train_X, train_y)
print("Best Parameters:", grid_search.best_params_)
print("Best F1-Score:", grid_search.best_score_)


# In[85]:


# Re-run the model with tuned parameters
# Best params obtained from grid search
best_params = {
    'C': 100,
    'penalty': 'l2',
    'solver': 'liblinear'
}

#  Logistic regression model
linear_clf = LogisticRegression(**best_params)
linear_clf.fit(train_X, train_y)

# Predictions and Evaluation
linear_predictions = linear_clf.predict(test_X)
print("Linear Classifier - Classification Report")
print(classification_report(test_y, linear_predictions))
print("Confusion Matrix")
print(confusion_matrix(test_y, linear_predictions))
# slightly better performance


# In[27]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# probability scores for the positive class
y_scores = linear_clf.predict_proba(test_X)[:, 1]

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(test_y, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# # SVM
# 

# In[54]:


# SVM Linear
svm = SVC(probability=True)
svm.fit(train_X, train_y)
svm_predictions = linear_svm.predict(test_X)
print("SVM - Classification Report")
print(classification_report(test_y, svm_predictions))
print("Confusion Matrix")
print(confusion_matrix(test_y, svm_predictions))


# In[30]:


# fine tune for using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],           # Regularization parameter
    'gamma': [0.001, 0.01, 0.1, 1],   # Kernel coefficient
    'kernel': ['linear', 'rbf']  # Different kernels
}
svm = SVC()
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='f1')
grid_search_svm.fit(train_X, train_y)
print("Best Parameters:", grid_search_svm.best_params_)
print("Best F1-Score:", grid_search_svm.best_score_)


# In[55]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

# Define the tuned SVM kernel
svm_tuned = SVC(kernel='rbf', C=100, gamma=0.1,probability=True)

# Perform cross-validation
cv_scores = cross_val_score(svm_tuned, train_X, train_y, cv=5, scoring='accuracy')

# Print cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

# Fit the tuned SVM model on the training data
svm_tuned.fit(train_X, train_y)

# Make predictions on the test data
svm_predictions = svm_tuned.predict(test_X)

# Evaluate the model's performance
print("RBF Kernel SVM - Classification Report")
print(classification_report(test_y, svm_predictions))
print("Confusion Matrix")
print(confusion_matrix(test_y, svm_predictions))


# In[56]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# cross-validation predictions
y_scores_cv = cross_val_predict(svm_tuned, train_X, train_y, cv=5, method="decision_function")

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(train_y, y_scores_cv)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# # Decision Tree Classifier

# In[35]:


#DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=seed)
decision_tree.fit(train_X, train_y)

decision_tree_predictions = decision_tree.predict(test_X)
decision_tree_report = classification_report(test_y, decision_tree_predictions)
decision_tree_cm = confusion_matrix(test_y, decision_tree_predictions)

print("Decision Tree - Classification Report")
print(decision_tree_report)
print("Confusion Matrix")
print(decision_tree_cm)


# In[36]:


from sklearn.tree import DecisionTreeClassifier
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

param_grid_decision_tree = {
    'criterion': ['entropy'],  # criterion
    'max_depth': [None, 5, 10, 15],  # Max depth of the tree
    'min_samples_split': [5, 10, 15, 20],  # Min number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 3],  # Min number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2', None] ,  # Number of features to consider when looking for the best split
}

decision_tree = DecisionTreeClassifier(random_state=seed)
grid_search_decision_tree = GridSearchCV(estimator=decision_tree, param_grid=param_grid_decision_tree, cv=5, scoring='f1')
grid_search_decision_tree.fit(train_X, train_y)

print("Best Parameters:", grid_search_decision_tree.best_params_)

print("Best F1-Score:", grid_search_decision_tree.best_score_)


# In[37]:


# tuned DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(criterion='entropy',
                                       max_depth=None,
                                       max_features=None,  
                                       min_samples_leaf=1,
                                       min_samples_split=5,
                                       random_state=seed)
decision_tree.fit(train_X, train_y)

decision_tree_predictions = decision_tree.predict(test_X)
decision_tree_report = classification_report(test_y, decision_tree_predictions)
decision_tree_cm = confusion_matrix(test_y, decision_tree_predictions)

print("Decision Tree - Classification Report")
print(decision_tree_report)
print("Confusion Matrix")
print(decision_tree_cm)

# improved f1-score, confusion matrix result


# In[38]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# probability scores for the positive class
y_scores = decision_tree.predict_proba(test_X)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(test_y, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Decision Tree')
plt.legend(loc="lower right")
plt.show()


# ## Random Forest
# 

# In[39]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(random_state=seed)
random_forest.fit(train_X, train_y)
rf_predictions = random_forest.predict(test_X)
print("Random Forest - Classification Report")
print(classification_report(test_y, rf_predictions))
print("Confusion Matrix")
print(confusion_matrix(test_y, rf_predictions))


# In[40]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_grid_random_forest = {
    'n_estimators': [100, 200, 300],           # Number of trees in the forest
    'criterion': ['gini', 'entropy'],          # Splitting criterion
    'max_depth': [None, 5, 10, 20],            # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],           # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],             # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2']   # Number of features to consider when looking for the best split
}
random_forest = RandomForestClassifier(random_state=seed)
grid_search_random_forest = RandomizedSearchCV(estimator=random_forest, param_distributions=param_grid_random_forest, n_iter=50, cv=5, scoring='f1')
grid_search_random_forest.fit(train_X, train_y)
print("Best Parameters:", grid_search_random_forest.best_params_)
print("Best F1-Score:", grid_search_random_forest.best_score_)


# In[41]:


best_params = {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 20, 'criterion': 'entropy'}
random_forest = RandomForestClassifier(**best_params)
random_forest.fit(train_X, train_y)
rf_predictions = random_forest.predict(test_X)
print("Random Forest - Classification Report")
print(classification_report(test_y, rf_predictions))
print("Confusion Matrix")
print(confusion_matrix(test_y, rf_predictions))

# Slight improve f1-score and precision for class 1


# In[42]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# probability scores for the positive class
y_scores = random_forest.predict_proba(test_X)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(test_y, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Random Forest')
plt.legend(loc="lower right")
plt.show()


# ## VotingClassifier using Soft Voting

# In[58]:


from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Initialize base estimators with appropriate settings
#linear_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
svm = SVC(kernel='rbf', probability=True)

# Create the VotingClassifier with soft voting
voting_clf = VotingClassifier(estimators=[
    
    ('rbf_svm', svm_tuned),
    ('dt', decision_tree),
    ('rf', random_forest)],
    voting='soft')

# Train the VotingClassifier
voting_clf.fit(train_X, train_y)

# Predict using the voting classifier
voting_predictions = voting_clf.predict(test_X)
print("Voting Classifier (soft Voting) - Classification Report")
print(classification_report(test_y, voting_predictions))
print("Confusion Matrix")
print(confusion_matrix(test_y, voting_predictions))


# In[59]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# probability scores for the positive class (class 1)
y_scores = voting_clf.predict_proba(test_X)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(test_y, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Voting Classifier')
plt.legend(loc="lower right")
plt.show()


# In[61]:


import joblib
joblib.dump(voting_clf, 'D:/centennialcollege/winter2024/Comp247/groupproject/voting_clf.pkl') 
joblib.dump(pipeline, 'D:/centennialcollege/winter2024/Comp247/groupproject/pipeline.pkl') 


# In[ ]:




