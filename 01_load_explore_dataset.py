import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

# converting to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("First 5 rows of the dataset:")
print(df.head())

# basic info
print("\nDataset Info:")
print(df.info())

# class distribution
print("\nTarget class distribution:")
print(df['target'].value_counts())
