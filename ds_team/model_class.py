import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class Model:
	
	def __init__(self,
		  model_type:str = 'random_forest',
                 ):
		self.model_type = model_type
		self.model = None

	def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
		"""
		This function will train a model by choice
            and save it as part of the class objects
		"""
		if self.model_type == 'random_forest':
			self.model = RandomForestRegressor()
			self.model.fit(x_train,y_train)

	def predict(self, x_test: pd.DataFrame)-> pd.DataFrame:
		"""
		This function will predict a model by choice
		  as pandas data frame
		"""
		if self.model_type == 'random_forest':
			x_test['predict']=self.model.predict(x_test)
			return x_test

