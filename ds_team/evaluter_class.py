import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

class Evaluator:
    
    def __init__(self
                 ):
        pass

    def evalute(self, y_test: pd.DataFrame , y_pred: pd.DataFrame):
        """
        This function will take the true labels and predictions
          and calculate various evaluation metrics
        """
        report = classification_report(y_test, y_pred)
        return report
