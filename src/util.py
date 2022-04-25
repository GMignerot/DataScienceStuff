import numpy as np


class Model (object):
	def predict(self, data):
		if data.ndim == 1:
			return self._predict_one(data)
		else:
			return np.asarray([self._predict_one(vector) for vector in data])

	def score(self, data, target):
		prediction = self.predict(data)
		return np.sum(prediction == target) / data.shape[0]