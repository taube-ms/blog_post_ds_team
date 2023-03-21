import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class ErrorAnalysis:
    
    def __init__(self):
        pass

    def plot_confusion_matrix(self, df_results: pd.DataFrame) -> None:
        """
        This function is plotting the confusion matrix result
        """
        mat_con = (confusion_matrix(df_results['y_true'], df_results['y_pred'], labels=[0,1,2]))
        # Setting the attributes
        fig, px = plt.subplots(figsize=(7.5, 7.5))
        px.matshow(mat_con, cmap=plt.cm.YlOrRd, alpha=0.5)
        for m in range(mat_con.shape[0]):
            for n in range(mat_con.shape[1]):
                px.text(x=m,y=n,s=mat_con[m, n], va='center', ha='center', size='xx-large')

        # Sets the labels
        plt.xlabel('Predictions', fontsize=16)
        plt.ylabel('Actuals', fontsize=16)
        plt.title('Confusion Matrix', fontsize=15)
        plt.show()
