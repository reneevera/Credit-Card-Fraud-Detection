# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc

# Load and explore the dataset
df = pd.read_csv('/creditcard.csv')
print("Rows & Columns:\n", df.shape)
print("\nColumn Names:\n", list(df.columns))
print("\nColumn Names and data types\n", df.dtypes)
print("0 = Normal Transaction, 1 = Fraud\n\n", df['Class'].value_counts())
print(round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset are normal transactions')
print(round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset are fraud transactions')
print("Missing Values in Dataset: ", df.isnull().sum().max())
print("Stats:\n", df.describe())

# Scaling 'Time' and 'Amount' using RobustScaler to reduce the impact of outliers
df['scaled_amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1,1))


# Drop the original 'Time' and 'Amount' columns and move the scaled versions to the first columns
df.drop(['Time', 'Amount'], axis=1, inplace=True)
df.insert(0, 'scaled_time', df.pop('scaled_time'))
df.insert(1, 'scaled_amount', df.pop('scaled_amount'))

# Visualize the dataset with updated features
sns.boxplot(x='Class', y='scaled_amount', data=df)
plt.title('Distribution of Scaled Transaction Amounts by Class')
plt.ylim(-5, 20)
plt.xticks([0, 1], ['Normal', 'Fraud'])
plt.xlabel('Class')
plt.ylabel('Scaled Amount')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df['scaled_time'], bins=50, alpha=0.7)
plt.xlabel('Scaled Transaction Time')
plt.ylabel('Frequency')
plt.title('Distribution of Scaled Transaction Time')
plt.show()

# Handle class imbalance using ADASYN
X = df.drop('Class', axis=1)
y = df['Class']
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# Split the data into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)


# Feature selection with the resampled dataset
selector = SelectKBest(f_classif, k=10)
train_X_new = selector.fit_transform(train_X, train_y)
best_features_indices = selector.get_support(indices=True)
print("Selected Features Indices:", best_features_indices)
train_X = pd.DataFrame(train_X_new, columns=train_X.columns[best_features_indices])


# Feature scaling
pipeline = Pipeline([
    ('scaling', StandardScaler())
])
pipeline.fit(train_X)
test_X_selected = pd.DataFrame(selector.transform(test_X), columns=train_X.columns)
test_X = pipeline.transform(test_X_selected)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
linear_clf = LogisticRegression()
linear_clf.fit(train_X, train_y)
linear_predictions = linear_clf.predict(test_X)
print("Linear Classifier - Classification Report")
print(classification_report(test_y, linear_predictions))
print("Confusion Matrix")
print(confusion_matrix(test_y, linear_predictions))

# Fine-tune Logistic Regression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, cv=5, scoring='f1')
grid_search.fit(train_X, train_y)
print("Best Parameters:", grid_search.best_params_)
print("Best F1-Score:", grid_search.best_score_)

best_params = {
    'C': 0.001,
    'penalty': 'l2',
    'solver': 'liblinear'
}
linear_clf = LogisticRegression(**best_params)
linear_clf.fit(train_X, train_y)
linear_predictions = linear_clf.predict(test_X)
print("Linear Classifier - Classification Report")
print(classification_report(test_y, linear_predictions))
print("Confusion Matrix")
print(confusion_matrix(test_y, linear_predictions))

# ROC curve for Logistic Regression
from sklearn.metrics import roc_curve, auc
y_scores = linear_clf.predict_proba(test_X)[:, 1]
fpr, tpr, _ = roc_curve(test_y, y_scores)
roc_auc = auc(fpr, tpr)
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

# Support Vector Machine (SVM)
from sklearn.svm import SVC
svm = SVC(probability=True)
svm.fit(train_X, train_y)
svm_predictions = svm.predict(test_X)
print("SVM - Classification Report")
print(classification_report(test_y, svm_predictions))
print("Confusion Matrix")
print(confusion_matrix(test_y, svm_predictions))

# Fine-tune SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}
grid_search_svm = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=5, scoring='f1')
grid_search_svm.fit(train_X, train_y)
print("Best Parameters:", grid_search_svm.best_params_)
print("Best F1-Score:", grid_search_svm.best_score_)

svm_tuned = SVC(kernel='rbf', C=100, gamma=0.1,probability=True)
cv_scores = cross_val_score(svm_tuned, train_X, train_y, cv=5, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())
svm_tuned.fit(train_X, train_y)
svm_predictions = svm_tuned.predict(test_X)
print("RBF Kernel SVM - Classification Report")
print(classification_report(test_y, svm_predictions))
print("Confusion Matrix")
print(confusion_matrix(test_y, svm_predictions))

# ROC curve for SVM
from sklearn.model_selection import cross_val_predict
y_scores_cv = cross_val_predict(svm_tuned, train_X, train_y, cv=5, method="decision_function")
fpr, tpr, thresholds = roc_curve(train_y, y_scores_cv)
roc_auc = auc(fpr, tpr)
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

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(train_X, train_y)
decision_tree_predictions = decision_tree.predict(test_X)
decision_tree_report = classification_report(test_y, decision_tree_predictions)
decision_tree_cm = confusion_matrix(test_y, decision_tree_predictions)
print("Decision Tree - Classification Report")
print(decision_tree_report)
print("Confusion Matrix")
print(decision_tree_cm)

# Fine-tune Decision Tree
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
param_grid_decision_tree = {
    'criterion': ['entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['auto', 'sqrt', 'log2', None]
}
grid_search_decision_tree = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid_decision_tree, cv=5, scoring='f1')
grid_search_decision_tree.fit(train_X, train_y)
print("Best Parameters:", grid_search_decision_tree.best_params_)
print("Best F1-Score:", grid_search_decision_tree.best_score_)

decision_tree = DecisionTreeClassifier(criterion='entropy',
                                       max_depth=None,
                                       max_features=None,
                                       min_samples_leaf=3,
                                       min_samples_split=5,
                                       random_state=42)
decision_tree.fit(train_X, train_y)
decision_tree_predictions = decision_tree.predict(test_X)
decision_tree_report = classification_report(test_y, decision_tree_predictions)
decision_tree_cm = confusion_matrix(test_y, decision_tree_predictions)
print("Decision Tree - Classification Report")
print(decision_tree_report)
print("Confusion Matrix")
print(decision_tree_cm)

# ROC curve for Decision Tree
y_scores = decision_tree.predict_proba(test_X)[:, 1]
fpr, tpr, thresholds = roc_curve(test_y, y_scores)
roc_auc = auc(fpr, tpr)
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

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(train_X, train_y)
rf_predictions = random_forest.predict(test_X)
print("Random Forest - Classification Report")
print(classification_report(test_y, rf_predictions))
print("Confusion Matrix")
print(confusion_matrix(test_y, rf_predictions))

# Fine-tune Random Forest
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_grid_random_forest = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
grid_search_random_forest = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42), param_distributions=param_grid_random_forest, n_iter=50, cv=5, scoring='f1')
grid_search_random_forest.fit(train_X, train_y)
print("Best Parameters:", grid_search_random_forest.best_params_)
print("Best F1-Score:", grid_search_random_forest.best_score_)

best_params = {'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'criterion': 'entropy'}
random_forest = RandomForestClassifier(**best_params)
random_forest.fit(train_X, train_y)
rf_predictions = random_forest.predict(test_X)
print("Random Forest - Classification Report")
print(classification_report(test_y, rf_predictions))
print("Confusion Matrix")
print(confusion_matrix(test_y, rf_predictions))

# ROC curve for Random Forest
y_scores = random_forest.predict_proba(test_X)[:, 1]
fpr, tpr, thresholds = roc_curve(test_y, y_scores)
roc_auc = auc(fpr, tpr)
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

# Voting Classifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
svm = SVC(kernel='rbf', probability=True)
voting_clf = VotingClassifier(estimators=[
    ('rbf_svm', svm_tuned),
    ('dt', decision_tree),
    ('rf', random_forest)],
    voting='soft')
voting_clf.fit(train_X, train_y)
voting_predictions = voting_clf.predict(test_X)
print("Voting Classifier (soft Voting) - Classification Report")
print(classification_report(test_y, voting_predictions))
print("Confusion Matrix")
print(confusion_matrix(test_y, voting_predictions))

# ROC curve for Voting Classifier
from sklearn.metrics import roc_curve, auc
y_scores = voting_clf.predict_proba(test_X)[:, 1]
fpr, tpr, thresholds = roc_curve(test_y, y_scores)
roc_auc = auc(fpr, tpr)
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