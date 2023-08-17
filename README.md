# blog_post_ds_team

The Data Science Collaboration Package 

This repo contains:
- Data Class
- Evaluator Class
- Model Class
- Features Class
- Experiment Class
- Error anaylsis class


## Using as a package

### Getting started

Install/clone the repo and import all the libraries.
Use the [implmentation python file](implmentation.py) or the [run notebook](run.ipynb) to run an end to end experiment.

### ds_team package

The daisl_ts python package contains logic for reproducible research. 
The main building blocks are:
1. `Data`: Provides a standardized approach for retrieving datasets to train and evaluate models.
2. `Model`: An abstract class that includes methods such as .fit and .predict which can be customized for specific model types.
3. `Features`: An abstract class that includes the .create_features method, which takes in a pandas dataframe and returns a modified dataframe with the necessary features for model training.
4. `Evaluator`: A set of evaluation metrics to be implemented based on the specific requirements of the project.
5. `Experiment`: A framework for conducting time-series cross-validation, running the chosen model, and measuring its performance.
6. `ErrorAnalysis`: A flexible class designed to accommodate additional functions for generating relevant plots or statistics.

### How to install an env in conda

Create the env:
```console
conda create --name <env_name> python=3.8 --file requirements.txt 
```

Activate the env:
```console
conda activate <env_name>
```
