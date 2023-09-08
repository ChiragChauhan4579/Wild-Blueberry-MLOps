# RTA-MLOps

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

Add the RTA_dataset file to your raw folder 

*Note* : Comment line 79 '/data/' in gitignore file because now the data is going to be tracked by DVC.

## Tracking the dataset with DVC

Run this command to track the raw data file `dvc add "data\raw\WildBlueberryPollinationSimulationData.csv"`. Upon completion you will find `WildBlueberryPollinationSimulationData.csv.dvc` file

Add dvc storage to dagshub using `dvc remote add path/to_dagshub.dvc`

using `dvc push -r "origin"`

## Scripts and MLflow tracking

Add scripts to src/data and src/models

