#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[11]:


# Loading dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
names = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
         'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
         'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
         'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
         'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
         'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
data = pd.read_csv(url, names=names)


# In[12]:


# Step 2: Data Preprocessing and Visualization
# Dropping unnecessary columns (id)
data.drop('id', axis=1, inplace=True)

# Plot distribution of diagnosis
plt.figure(figsize=(6, 4))
sns.countplot(x='diagnosis', data=data)
plt.title('Diagnosis Distribution')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.show()

# Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Split dataset into features (X) and target variable (y)
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


# Step 3: Model Selection
# Initializing Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)


# In[14]:


# Step 4: Model Training
# Training the model on the training data
rf_model.fit(X_train, y_train)


# In[16]:


# Step 5: Model Evaluation
# Predict on the testing data
y_pred = rf_model.predict(X_test)
# Evaluate the model
print(classification_report(y_test, y_pred))


# In[ ]:




