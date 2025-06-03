import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

# normalizing features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

print("First 5 rows of normalized features:")
print(X_normalized[:5])
