import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load the dataset
df = pd.read_csv('survey.csv')

# Remove irrelevant columns
df.drop(columns=['Timestamp', 'state', 'comments'], inplace=True)

# Normalize gender values
def clean_gender(g):
    g = g.strip().lower()
    if g in ['male', 'm', 'man']:
        return 'Male'
    elif g in ['female', 'f', 'woman']:
        return 'Female'
    else:
        return 'Other'

df['Gender'] = df['Gender'].apply(clean_gender)

# Save original gender labels for plotting
df['Gender_Original'] = df['Gender']

# Remove unrealistic age values
df = df[(df['Age'] >= 16) & (df['Age'] <= 100)]

# Fill missing values
df = df.copy()
df['self_employed'] = df['self_employed'].fillna("Don't know")
df['work_interfere'] = df['work_interfere'].fillna("Don't know")

# Select useful features for prediction
features = [
    'Age', 'Gender', 'self_employed', 'family_history', 'work_interfere',
    'no_employees', 'remote_work', 'tech_company', 'benefits',
    'care_options', 'wellness_program', 'seek_help', 'anonymity',
    'leave', 'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_health_interview',
    'phys_health_interview', 'mental_vs_physical', 'obs_consequence'
]
df = df[features + ['treatment', 'Gender_Original']]

# Encode categorical columns to numbers
le = LabelEncoder()
for col in features + ['treatment']:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Split into features (X) and target (y)
X = df.drop(['treatment', 'Gender_Original'], axis=1)
y = df['treatment']

# Create train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tp = cm[1][1]
tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Add a new age group column
df['age_group'] = pd.cut(df['Age'], bins=[15, 25, 35, 50, 100], labels=['16–25', '26–35', '36–50', '51+'])
actual_counts = np.bincount(y_test)
pred_counts = np.bincount(y_pred)
error_labels = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
error_values = [tp, tn, fp, fn]
importances = model.feature_importances_
feat_names = X.columns
imp_df = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(6)

# --- DISPLAY SEPARATE CHARTS ONE BY ONE WITH EXPLANATIONS ---

# Pie chart: prediction class proportions
plt.figure(figsize=(5, 5))
colors = ['#AED6F1', '#58D68D']
plt.pie(pred_counts, labels=['Not Seeking Treatment', 'Seeking Treatment'], autopct='%1.1f%%', startangle=90, colors=colors)
plt.title("Model's Predicted Distribution of Mental Health Treatment")
plt.legend(labels=['Not Seeking Treatment (Blue)', 'Seeking Treatment (Green)'], loc='upper right')
plt.show()

# Bar chart: actual test distribution
plt.figure(figsize=(5, 5))
bars = plt.bar(['Not Seeking', 'Seeking'], actual_counts, color=colors)
plt.title("Actual Distribution of Treatment in Test Data")
plt.ylabel("Number of Individuals")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, yval, ha='center', fontsize=10)
plt.legend(bars, ['Not Seeking (Blue)', 'Seeking (Green)'], loc='upper right')
plt.show()

# Age group vs treatment
plt.figure(figsize=(6, 4))
sns.countplot(x='age_group', hue='treatment', data=df, palette=colors)
plt.title("Treatment Seeking by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Individuals")
plt.legend(title="Treatment", labels=['Not Seeking', 'Seeking'])
plt.show()

# Gender vs treatment (using original labels)
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender_Original', hue='treatment', data=df, palette=colors)
plt.title("Treatment Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Individuals")
plt.legend(title="Treatment", labels=['Not Seeking', 'Seeking'])
plt.show()

# Work interfere vs treatment
plt.figure(figsize=(6, 4))
sns.countplot(y='work_interfere', hue='treatment', data=df, palette=colors)
plt.title("Does Work Interference Affect Treatment Seeking?")
plt.xlabel("Number of Individuals")
plt.ylabel("Work Interfere Response")
plt.legend(title="Treatment", labels=['Not Seeking', 'Seeking'])
plt.show()

# Feature importance bar plot
plt.figure(figsize=(6, 4))
sns.barplot(x=imp_df.values, y=imp_df.index, palette='Blues_r')
plt.title("Top Features Influencing the Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
for index, value in enumerate(imp_df.values):
    plt.text(value + 0.002, index, f"{value:.2f}", va='center')
plt.show()

# Confusion matrix error types
plt.figure(figsize=(6, 4))
colors_error = ['#5DADE2', '#45B39D', '#F1948A', '#C39BD3']
sns.barplot(x=error_labels, y=error_values, palette=colors_error)
plt.title("Prediction Breakdown: TP, TN, FP, FN")
plt.ylabel("Number of Cases")
for i, v in enumerate(error_values):
    plt.text(i, v + 1, str(v), ha='center')
plt.legend(handles=[
    plt.Rectangle((0,0),1,1, color=c) for c in colors_error
], labels=error_labels, title='Type', loc='upper right')
plt.show()