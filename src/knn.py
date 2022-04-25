import heapq
import numpy as np
from util import Model


class kNN (Model):
	def __init__(self, data, target, k=1):
		self.k = k
		self._fit(data, target)

	def _fit(self, data, target):
		self.data = data
		self.target = target

	def _predict_one(self, vector):
		nearest = []
		for i in range(self.data.shape[0]):
			datavector = self.data[i]
			proximity = -np.linalg.norm(datavector - vector)
			if len(nearest) < self.k:
				heapq.heappush(nearest, (proximity, self.target[i]))
			elif proximity > nearest[0][0]:
				heapq.heapreplace(nearest, (proximity, self.target[i]))

		most_frequent = max(set(nearest), key=nearest.count)[1]
		return most_frequent


	def _distance(self, v1, v2):
		return np.linalg.norm(v2 - v1)



if __name__ == "__main__":
	from sklearn.datasets import load_iris
	from sklearn.model_selection import train_test_split

	iris = load_iris()
	X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
	
	for k in range(1, 10):
		model = kNN(X_train, y_train, k=k)
		print(f"With k={k} :\nTraining set : {model.score(X_train, y_train)}\nTesting set : {model.score(X_test, y_test)}\n")