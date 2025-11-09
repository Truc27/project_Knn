import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
data = pd.read_csv("dataset.csv")
X = data.drop(columns=["y"]).values
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

    def distance_table(self, new_point):
        distances = [euclidean_distance(point, new_point)
                     for point in self.X_train]
        distances_table = pd.DataFrame({
            'X1': self.X_train[:, 0],
            'X2': self.X_train[:, 1],
            'Label': self.y_train,
            'Distance': distances
        }).sort_values(by='Distance', ascending=True).reset_index(drop=True)
        print("Bảng khoảng cách:")
        print(distances_table)
        print("-" * 50)
        return distances_table


knn = KNN(int(input("nhap k: ")))
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = np.mean(predictions == y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")
p1 = np.array([float(x) for x in input("Nhap diem can tim 1: ").split(', ')])
new_points = np.array([p1])
new_predictions = knn.predict(new_points)
lable1 = knn.distance_table(p1)
print("\n Dự đoán cho hai điểm:")
print(f"Điểm {p1} thuộc lớp: {new_predictions[0]}")
