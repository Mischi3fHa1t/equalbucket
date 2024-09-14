"""
Decision Tree Classifier
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Seed for reproducibility
np.random.seed(0)

# Number of samples
NSAMPLES = 1000

# Feature weights
weights = [0.9, 0.04, 0.03, 0.03]
features = ['MATCH', 'REDUCE', 'LOWCVE', 'OTHER']

# Create synthetic data with weighted features
data = {
    'MATCH': np.random.uniform(0, 1, NSAMPLES),
    'REDUCE': np.random.uniform(0, 1, NSAMPLES),
    'LOWCVE': np.random.uniform(0, 1, NSAMPLES),
    'OTHER': np.random.uniform(0, 1, NSAMPLES),
}

# Calculate feature importance influence
X = pd.DataFrame(data)
X_weighted = X.apply(lambda x: x * weights, axis=1)

# Simulate binary output based on feature weights
# Weighted sum of features
weighted_sum = X_weighted.sum(axis=1)
# Binary output with probability of scanning influenced by the weighted sum
probability_scan = 1 / (1 + np.exp(-weighted_sum))
data['Scan'] = np.random.binomial(1, probability_scan)

# Create DataFrame with target
df = pd.DataFrame(data)

# Define features and target variable
X = df[['MATCH', 'REDUCE', 'LOWCVE', 'OTHER']]
y = df['Scan']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Display feature importance
print("\nFeature Importances:")
for feature, importance in zip(features, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")
