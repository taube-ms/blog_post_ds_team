import pandas as pd
from sklearn.model_selection import train_test_split

from ds_team.data_class import Data
from ds_team.evaluter_class import Evaluator
from ds_team.features_class import Features
from ds_team.model_class import Model

class Experiment:
    
    def __init__(self,
          data_class: Data,
          features_class: Features,
          model_class: Model,
          evaluator_class: Evaluator,
          target: str,
          features: list,
          split: float = 0.3):
        
        self.data_class = data_class
        self.features_class = features_class
        self.model_class = model_class
        self.evaluator_class = evaluator_class
        self.target = target
        self.features = features
        self.split = split

    def run(self) -> list:
        """
        This function will run the experiment and return 
          the evaluation metrics and the data frame with
          the true and predicted labels
        """
        df = self.data_class.get_data()
        df_features = self.features_class.create_features(df)
        X = df_features[self.features]
        y = df_features[self.target].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split, random_state=42)
        self.model_class.fit(X_train, y_train)
        y_predict = self.model_class.predict(pd.DataFrame(X_test))['predict'].astype(int)
        final_result = self.evaluator_class.evalute(y_test, y_predict)
        df_result = X_test
        df_result['y_pred'] = y_predict
        df_result['y_true'] = y_test
        return final_result, df_result
