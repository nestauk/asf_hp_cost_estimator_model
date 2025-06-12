# 🏡 Air source heat pump cost estimator 🏡

The `asf_hp_cost_estimator_model` repository contains the code to model and predict the cost of an air source heat pump:

- in residential properties
- installed as part of a retrofit (heat pumps installed in new builds or as part of a cluster of installations are excluded)
- in houses or bungalows (flats are excluded from the analysis)
- houses with 2 or more [habitable rooms](https://epc.opendatacommunities.org/docs/guidance#field_domestic_NUMBER_HABITABLE_ROOMS) and with a floor area between 20 and 500 m2
- for Scotland, Wales and English regions

## 🚀 Modelling the cost of an air source heat pump

[Quantile regression gradient boosting regressor models](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html) are fitted to create prediction intervals for the cost of an air source heat pump (80% confidence intervals, by fitting models on the 10th and 90th percentile).

The target variable is the overall cost of installation and the predictors include:

- Total floor area
- Number of habitable rooms (2 to 8+)
- Number of days between 2007 and HP installation (as a measure of time)
- Property built form: detached, semi detached, mid terrace and end terrace
- Property type: bungalow and house
- Construction age band: pre-1929, 1930-1965, 1966-1982, 1983-2006 and 2007 onwards
- Region: Scotland, Wales, London, East Midlands, West Midlands, East of England, South East, South West, North West, North East and Yorkshire and the Humber.

## 🆕 Latest data

The latest model in use by the cost estimator tool was trained on data up to **Q1 2025** (March 2025).

## 🧩 Data sources

### Microgeneration Certification Scheme (MCS) data on heat pump installations

This is a subset of the [MCS Installations Database (MID)](https://certificate.microgenerationcertification.org/), and contains one record for each MCS certificate associated with a heat pump installation. The dataset contains records of both domestic and non-domestic air source, water/ground source and other types of heat pump installations. Features in the dataset include:

- information about the property: address, heat and water demand
- characteristics of the heat pump installed: type, model, manufacturer, capacity, flow temperature, SCOP
- information about the installation: commissioning date, overall cost of installation

The overall installation cost is the full cost of installation including materials and labour, not just the cost of the heat pump unit. To note that this cost is the cost prior to deducting government grants such as the Boiler Upgrade Scheme (BUS) grant or Home Energy Scotland (HES) grant.

MID data is used with permission from MCS and subject to the conditions of a data sharing agreement.

## Energy Performance Certificates (EPC) register data about homes

Property data comes from [England and Wales](https://epc.opendatacommunities.org/) and [Scotland's](https://statistics.gov.scot/resource?uri=http%3A%2F%2Fstatistics.gov.scot%2Fdata%2Fdomestic-energy-performance-certificates) EPC register. The EPC register provides data on building characteristics and energy efficiency measures, including:

- Property address and other location information;
- Property characteristics such as number of rooms, property type and built form.
- Heating system(s) installed;
- Energy efficiency ratings.

The EPC Register datasets are open-source and accessible to everyone.

## Additional data

### Location lookups

The following location lookups are used:

- [Postcode to OA (2021) to LSOA to MSOA to LAD (November 2024) Best Fit Lookup in the UK](https://open-geography-portalx-ons.hub.arcgis.com/datasets/068ee476727d47a3a7a0d976d4343c59/about)

- [Local Authority District to Region (December 2024) Lookup in EN](https://geoportal.statistics.gov.uk/datasets/ons::local-authority-district-to-region-december-2024-lookup-in-en/about)

- [Postcode to OA (2021) to LSOA to MSOA to LAD (November 2024) Best Fit Lookup in the UK](https://open-geography-portalx-ons.hub.arcgis.com/datasets/068ee476727d47a3a7a0d976d4343c59/about)

- [Local Authority District to Region (April 2021) Lookup in EN](https://geoportal.statistics.gov.uk/datasets/ons::local-authority-district-to-region-april-2021-lookup-in-en/about)

### Inflation and price indices

The "CPI INDEX 05.3 : Household appliances, fitting and repairs 2015=100" from the inflation and price indices data was [sourced from the ONS](https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/d7ck/mm23)

## ⚒️ Data processing & joining

The underlying dataset used to model the cost of an air source installation is the MCS installations dataset enhanced with EPC information about properties. MCS and EPC datasets are cleaned and preprocessed before being joined. Installations without EPC property information are removed from the analysis. The code for preprocessing and joining MCS to EPC is available in the [asf_core_data GitHub repository](https://github.com/nestauk/asf_core_data).

## 🗂️ Repository structure

The repository structure and key scripts are highlighted below:

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
│    ├─ data_processing/ - further data processing prior to modelling
|    |   ├─ process_installations_data.py
│    ├─ model_training/ - model training scripts
|    |    |- fit_cost_prediction_intervals.py
│    ├─ model_evaluation/ - scripts for model evaluation
|    |    |- cross_validation.py
│    ├─ hyperparameter_tuning/ - scripts for hyperparameter tuning
|    |    |- tune_hyperparameters.py
│    ├─ README.md - instructions to run the different pipelines
├───utils/
│    Utils for plotting and evaluation
│    ├─ plotting_utils.py
│    ├─ model_evaluation_utils.py
├───notebooks/
│    Notebooks for data and model exploration

```

## 📋 Instructions for retraining the model

These are instructions for data scientists at Nesta.

When new quarter data is made available you can follow the steps to retrain the cost models (after the data has been processed with [asf_core_data](https://github.com/nestauk/asf_core_data)).

1. Open an issue in this GitHub repository, such as "Retrain model with QX 202Y data"
2. Update `asf_hp_cost_estimator_model/config/base.yaml`
   - `cpi_reference_year`: update the CPI reference year accordingly
   - Location data sources: review and update location sources as required
   - `mcs_epc_filename_date`: update with newest date of MCS-EPC data processing
3. Re-run hyperparameter tuning pipeline:

- Run `python asf_hp_cost_estimator_model/pipeline/hyperparameter_tuning/tune_hyperparameters.py`
- Take note of the hyperparameters logged

4. Update `asf_hp_cost_estimator_model/config/base.yaml` after tuning hyperparameters:
   - change `hyper_parameters` according to the hyperparameters logged in the previous step
5. Re-run cross-validation pipeline:
   - Run `python asf_hp_cost_estimator_model/pipeline/model_evaluation/cross_validation.py`
   - Assess results logged
6. Retrain models:
   - Run `python asf_hp_cost_estimator_model/pipeline/model_training/fit_cost_prediction_intervals.py`
   - Models are saved to S3
7. Update sections "🆕 Latest data" and "🧩 Data sources" of this `REAMDE.md` to reflect changes.
8. Let the tech/design team know that the model has been updated, so that they can restart the API.

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
