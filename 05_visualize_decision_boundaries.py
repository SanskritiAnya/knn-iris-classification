import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data[:, 2:4]  # petal length and petal width
y = iris.target

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_normalized, y)

# plotting decision boundaries
h = 0.02  # step size in mesh
x_min, x_max = X_normalized[:, 0].min() - 1, X_normalized[:, 0].max() + 1
y_min, y_max = X_normalized[:, 1].min() - 1, X_normalized[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel("Petal Length (normalized)")
plt.ylabel("Petal Width (normalized)")
plt.title("Decision Boundaries - KNN (k=3)")
plt.grid(True)
plt.tight_layout()
plt.show()
