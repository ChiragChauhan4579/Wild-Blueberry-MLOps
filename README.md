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
