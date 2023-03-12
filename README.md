# artificial-neural-network
A simple implementation for Artificial Neural Networks.

## Dependencies
NumPy

## Usage Example
```python
import numpy as np

from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import deep_learning as dl

# Load the iris dataset
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target.reshape(-1, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Reshape the input matrix: shape=(no_features, no_examples) and encode the target
X_train = X_train.T
X_test = X_test.T
ohe = OneHotEncoder(sparse_output=False)
Y_train = ohe.fit_transform(y_train).T

# Initialize the NeuralNetwork with the following architecture
# Input_Layer: (4, tanh), Hidden_Layer: (4, tanh), Output_Layer: (3, sigmoid)
nn = dl.NeuralNetwork(n=[4, 4, 3], activations=['tanh', 'tanh', 'sigmoid'])

# Initialize a trainer with learning rate=0.03
t = dl.Trainer(alpha=0.01)

# Train the nn 
no_iterations = 3500
for i in range(no_iterations):
    t.train(nn, X_train, Y_train)

# Make Predictions
Y_pred = nn.for_prop(X_test)
y_pred = np.argmax(Y_pred, axis=0).reshape(-1, 1)

# Compute the accuracy
print(classification_report(y_test, y_pred))
```
Output: 
```
        precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       1.00      0.89      0.94        18
           2       0.85      1.00      0.92        11

    accuracy                           0.96        45
   macro avg       0.95      0.96      0.95        45
weighted avg       0.96      0.96      0.96        45
```

**Notice that sklearn is only used for importing the dataset, data preparation and classification report**
