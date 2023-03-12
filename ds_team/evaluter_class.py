import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

class Evaluator:
	
	def __init__(self
                 ):
		pass

	def evalute(self, y_test: pd.DataFrame , y_pred: pd.DataFrame) -> dict:
		"""
		This function will take the pred and test data and calculate the metrics.
		"""
		report = classification_report(y_test, y_pred)
		return report