# 🏡 Air source heat pump cost estimator 🏡

The `asf_hp_cost_estimator_model` repository contains the code to model and predict the cost of an air source heat pump:
- in the residential properties
- as part of a retrofit (i.e. new builds excluded)

## 🧩 Data sources

## 🆕 Latest data


## 🗂️ Repository structure

Below we have the repository structure and we highlight a few of the key folders and scripts:

```
asf_hp_cost_estimator_model
├───config/
│    Configuration scripts
│    ├─ base.yaml
├───getters/
│    Scripts with functions to load data from S3
│    ├─ data_getters.py
├───pipeline/
│    Subdirs with scripts to process data and produce outputs
│    ├─ data_processing/ - data processing prior to modelling
|    |   ├─ process_installations_data.py
│    ├─ model_training/ - model training scripts
|    |    |- fit_cost_model.py - 
|    |    |- fit_residuals.py -
│    ├─ model_evaluation/ - scripts for model evaluation
|    |    |- assess_feature_importance.py - 
|    |    |- conduct_cross_validation.py -
|    |    |- analyse_residuals.py -
|    |    |- residuals_model_cross_validation.py -
│    ├─ hyperparameter_tuning/ - scripts for data and model hyperparameter tuning
|    |    |- params_to_tune.py - 
|    |    |- hyperparameter_tuning.py -
│    ├─ README.md - instructions to run the different pipelines
├───utils/
│    ├─ plotting_utils.py -
```

## 📋 Instructions for retraining the model

## ⚙️ Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `direnv` and `conda`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure `pre-commit`

## 📢 Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
