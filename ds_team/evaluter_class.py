import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

class Evaluator:
    
    def __init__(self):
        self.metrics = {}
    
    def evalute(self, y_test: pd.DataFrame , y_pred: pd.DataFrame):
        """
        This function will take the true labels and predictions
          and calculate various evaluation metrics
        """
        report = classification_report(y_test, y_pred)
        
        # Calculate individual metrics and store in self.metrics
        self.metrics["precision"] = precision_score(y_test, y_pred, average="weighted")
        self.metrics["recall"] = recall_score(y_test, y_pred, average="weighted")
        self.metrics["f1_score"] = f1_score(y_test, y_pred, average="weighted")
        self.metrics["accuracy"] = accuracy_score(y_test, y_pred)
        
        return report
