# Wild-Blueberry-MLOps (MLflow + DVC) [Adding Evidently, Github actions, FastAPI deployment]

Tools used in the project:

* Cookiecutter: Data science project structure
* Data version control (DVC): Version control of the data assets and to make pipeline
* Github: For code version control
* GitHub Actions: To create the CI-CD pipeline
* MLFlow: For model tracking and registry
* Render: To deploy the application
* FastAPI: To create a web app
* EvidentlyAI: To evaluate and monitor ML models in production
* Pytest: To implement the unit tests

**Note**: Remember to change paths

Creating a new environment would be better to avoid any issues with dependencies

To get started with this run these commands:

```
python -m venv mlops_stack or conda create -n mlops_stack
activate the environment using mlops_stack\Scripts\activate or conda activate mlops_stack
pip install -r requirements.txt
```

## Creating the project structure with cookiecutter

Run the following command and fill necessary information

```
cookiecutter https://github.com/drivendata/cookiecutter-data-science
```

Now move inside your directory and push this to your github repository
```
git init 
git add . 
git commit -m "Adding cookiecutter template"
git remote add origin <your_github_repo>
git branch -M main
git push -u origin main
```

Add the necessary dataset file to your raw folder 

*Note* : Comment line 79 '/data/' in gitignore file because now the data is going to be tracked by DVC.

## Tracking the dataset with DVC

Run this command to track the raw data file `dvc add "data\raw\WildBlueberryPollinationSimulationData.csv"`. Upon completion you will find `WildBlueberryPollinationSimulationData.csv.dvc` file

Add dvc storage to dagshub using `dvc remote add path/to_dagshub.dvc`

using `dvc push -r "origin"`

## Scripts and MLflow tracking

Add scripts to src/data and src/models

In data data loading data(if there is any preprocessing at your end you can add it) else splitting data script are be executed directly

In models training and model selection script is added

From the model selection script the latest model will be saved in models folder of the root directory

## Adding DVC pipeline

Few arguments to look at before running.
* he -n switch gives the stage a name.
* The -d switch passes the dependencies to the command.
* The -o switch defines the outputs of the command.
* The -M switch defines the metrics of the command

Remember to npt allow tracking your data files by git. If you run below commands you will find errors, so just do the mentioned thing in the error and run again.

`dvc stage add --run --force -n data -d C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/src/data/split_data.py -d C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/data/raw -o C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/data/processed/wb_train.csv -o C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/data/processed/wb_test.csv python C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/src/data/split_data.py`

`dvc stage add --run --force -n train -d C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/src/models/train_model.py -d C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/data/processed/wb_train.csv -d C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/data/processed/wb_test.csv -p xgboost.max_depth -p xgboost.n_estimators python C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/src/models/train_model.py  --config=C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/params.yaml`

`dvc stage add --run --force -n evaluate -d C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/src/models/model_selection.py -d C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/src/models/mlruns/872714289021200779 -o C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/models/model.joblib python C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/src/models/model_selection.py`

Force is added in the command so if there is error and when you solve it you can force change the yaml file.

This pipeline executes only when the stage if dependencies are changed

To rerun the pipeline run `dvc repro`

## Future updates

1. Creating an API with Flask/FastAPI
2. Deploy on Render with CI/CD with GitHub actions
3. Add Evidently for monitoring