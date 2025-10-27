import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
data = pd.read_csv("dataset.csv")
X = data.drop(columns=["sample", "y"]).values
y = data["y"].values
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

new_points = np.array([[8, 1.5], [4.5, 7]])
new_predictions = knn.predict(new_points)
plt.scatter(X_train[y_train == 'B', 0], X_train[y_train == 'B', 1],
            color='tab:green', label='chu B')
plt.scatter(X_train[y_train == 'A', 0], X_train[y_train == 'A', 1],
            color='tab:red', label='chu A')
plt.scatter(new_points[:, 0], new_points[:, 1],
            color='tab:olive', label='Điểm dự đoán')
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()

print("\n Dự đoán cho hai nhân vật màu vàng:")
print(f"Điểm (8, 1.5) thuộc lớp: {new_predictions[0]}")
print(f"Điểm (4.5, 7) thuộc lớp: {new_predictions[1]}")
