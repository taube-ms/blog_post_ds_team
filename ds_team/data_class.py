import pandas as pd
import numpy as np
from sklearn import datasets

class Data:
    
    def __init__(self,
                data_version: str='iris'):
        self.data_version = data_version

    def get_data(self) -> pd.DataFrame:
        """
        This function will return the target table
          as pandas data frame
        """
        if self.data_version == 'iris':
            iris = datasets.load_iris() # load the dataset
            df = pd.DataFrame(
                data= np.c_[iris['data'], iris['target']],
                columns= iris['feature_names'] + ['target']
            )
        else:
            df = pd.DataFrame()
        return df
