import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
data = pd.read_csv("Data_iris.csv")
X = data.drop(columns=["class"]).values
y = data["class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=2)


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, new_points):
        predictions = [self.predict_class(new_point)
                       for new_point in new_points]
        return np.array(predictions)

    def predict_class(self, new_point):
        distances = [euclidean_distance(point, new_point)
                     for point in self.X_train]
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common


knn = KNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = np.mean(predictions == y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")

plt.scatter(X_train[y_train == 'Iris-setosa', 0],
            X_train[y_train == 'Iris-setosa', 1],
            color='tab:green', label='Iris-setosa')
plt.scatter(X_train[y_train == 'Iris-versicolor', 0],
            X_train[y_train == 'Iris-versicolor', 1],
            color='tab:red', label='Iris-versicolor')
plt.scatter(X_train[y_train == 'Iris-virginica', 0],
            X_train[y_train == 'Iris-virginica', 1],
            color='tab:olive', label='Iris-virginica')
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.legend()
plt.show()
