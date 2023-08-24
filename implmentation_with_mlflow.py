import mlflow
import mlflow.sklearn
from ds_team.data_class import Data
from ds_team.evaluter_class import Evaluator
from ds_team.features_class import Features
from ds_team.model_class import Model
from ds_team.experiment_class import Experiment
from ds_team.error_analysis_class import ErrorAnalysis

# Create instances of each class
my_data = Data(data_version='iris')
my_features = Features()
my_model = Model(model_type='random_forest')
my_evaluator = Evaluator()

# Define the parameters for the experiment
target = 'target'
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
split = 0.3

# Start an MLflow run
with mlflow.start_run():
    
    # Log parameters
    mlflow.log_param("data_version", 'iris')
    mlflow.log_param("model_type", 'random_forest')
    mlflow.log_param("target", target)
    mlflow.log_param("features", features)
    mlflow.log_param("split", split)

    # Create an instance of the Experiment class
    my_experiment = Experiment(data_class=my_data,
                               features_class=my_features,
                               model_class=my_model,
                               evaluator_class=my_evaluator,
                               target=target,
                               features=features,
                               split=split)

    # Run the experiment
    report, df_result = my_experiment.run()

    # Log the trained model
    mlflow.sklearn.log_model(my_model.model, "model")
    
    #  Log metrics from the evaluator
    for metric_name, metric_value in my_evaluator.metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    # Print the evaluation metrics
    print(report)
