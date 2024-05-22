""" Load datasets and models for testing purposes.
"""

import pandas as pd
import os
import pickle
import numpy as np

from sklearn.model_selection import train_test_split

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)

RANDOM_STATE = 1498672

IRIS = "Iris"
TITANIC = "Titanic"
SPACESHIP_TITANIC = "Spaceship-Titanic"
CALIFORNIA_HOUSING = "California-Housing"
OPENML_DIABETES = "OpenML-Diabetes"
ADULT = "Adult-Income"
KAGGLE_FLOOD = "Kaggle-Flood"
KAGGLE_HEART_FAILURE = "Kaggle-Heart-Failure"
BANK_CHURN = "Kaggle-Bank-Churn"
CANCER = "Wisconsin-Cancer"

DATASETS = [
    CALIFORNIA_HOUSING,
    OPENML_DIABETES,
    IRIS,
    TITANIC,
    SPACESHIP_TITANIC,
    ADULT,
    KAGGLE_FLOOD,
    KAGGLE_HEART_FAILURE,
    BANK_CHURN,
    CANCER,
]


def get_avaialble_datasets():
    return DATASETS


def get_dataset(dataset_name):
    """
    Loads the dataset with the given name and return the train and test splits.

    Returns: X_train, X_test, y_train, y_test, feature_names : 4x np.array, str
    """
    if dataset_name == SPACESHIP_TITANIC:
        df = pd.read_csv("../data/spaceship-titanic-train.csv")
        # transform cabin since 8000 unique values do not fit into the context windows of the LLM we want to use
        df["Cabin"] = df["Cabin"].map(
            lambda x: x[:1] + "/" + x[-1] if isinstance(x, str) else x
        )
        X_data = df.drop(columns=["PassengerId", "Transported", "Name"]).values
        y_data = df["Transported"].values
        feature_names = df.drop(
            columns=["PassengerId", "Transported", "Name"]
        ).columns.tolist()
    elif dataset_name == TITANIC:
        df = pd.read_csv("../data/titanic-train.csv")
        # drop rows with missing values
        df = df.dropna()
        y_data = df["Survived"].values
        df = df.drop(columns=["PassengerId", "Name", "Survived", "Cabin", "Ticket"])
        X_data = df.values
        feature_names = df.columns.tolist()
    elif dataset_name == IRIS:
        df = pd.read_csv("../data/iris.csv")
        df = df.dropna()
        y_data = df["species"].values == "Iris-setosa"  # binary classification
        df = df.drop(columns=["species"])
        X_data = df.values
        feature_names = df.columns.tolist()
    elif dataset_name == CALIFORNIA_HOUSING:
        df = pd.read_csv("../data/california-housing.csv")
        df = df.dropna()  # drop rows with missing values
        y_data = df["median_house_value"].values
        df = df.drop(columns=["median_house_value"])
        X_data = df.values
        feature_names = df.columns.tolist()
    elif dataset_name == OPENML_DIABETES:
        df = pd.read_csv("../data/openml-diabetes.csv")
        # drop rows with missing values
        df = df.dropna()
        y_data = df["Outcome"].values
        df = df.drop(columns=["Outcome"])
        X_data = df.values
        feature_names = df.columns.tolist()
    elif dataset_name == ADULT:
        df = pd.read_csv("../data/adult-train.csv")
        # drop fnlwgt
        df = df.drop(columns=["fnlwgt"])
        y_data = df["Income"].values == " >50K"
        df = df.drop(columns=["Income"])
        # convert categorical columns to numbers
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("category").cat.codes
        # drop na and inf values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        X_data = df.values
        feature_names = df.columns.tolist()
    elif dataset_name == KAGGLE_FLOOD:
        df = pd.read_csv("../data/kaggle-flood-train.csv")
        df = df.dropna()
        # subset 10 000 observations
        df = df.sample(n=10000, random_state=RANDOM_STATE)
        y_data = df["FloodProbability"].values
        df = df.drop(columns=["FloodProbability"])
        X_data = df.values
        feature_names = df.columns.tolist()
    elif dataset_name == KAGGLE_HEART_FAILURE:
        df = pd.read_csv("../data/kaggle_heart_failure_clinical_records.csv")
        df = df.dropna()
        y_data = df["DEATH_EVENT"].values
        df = df.drop(columns=["DEATH_EVENT"])
        X_data = df.values
        feature_names = df.columns.tolist()
    elif dataset_name == BANK_CHURN:
        df = pd.read_csv("../data/kaggle-bank-churn.csv")
        df = df.dropna()
        # drop surname feature
        df = df.drop(columns=["Surname"])
        # subset 10 000 observations
        df = df.sample(n=10000, random_state=RANDOM_STATE)
        y_data = df["Exited"].values == 0  # binary classification
        df = df.drop(columns=["Exited"])
        # drop na and inf values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        X_data = df.values
        feature_names = df.columns.tolist()
    elif dataset_name == CANCER:
        df = pd.read_csv("../data/Wisconsin-cancer.csv")
        # ddrop 'Unnamed: 32'
        df = df.drop(columns=["Unnamed: 32"])
        y_data = df["diagnosis"].values == "M"
        df = df.drop(columns=["diagnosis"])
        X_data = df.values
        feature_names = df.columns.tolist()
    else:
        raise ValueError("Unknown dataset: ", dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, feature_names


def get_ebm(dataset_name):
    """
    Returns: An EBM trained on the dataset.
    """
    X_train, X_test, y_train, y_test, feature_names = get_dataset(dataset_name)
    model_file = f"../models/{dataset_name}"
    if os.path.isfile(model_file):  # check if model file exists
        # load with pickle
        with open(model_file, "rb") as file:
            ebm = pickle.load(file)
    else:  # otherwise train and save
        if dataset_name in [
            SPACESHIP_TITANIC,
            IRIS,
            TITANIC,
            OPENML_DIABETES,
            ADULT,
            KAGGLE_HEART_FAILURE,
            BANK_CHURN,
            CANCER,
        ]:
            # classification
            ebm = ExplainableBoostingClassifier(
                interactions=0,
                feature_names=feature_names,
                random_state=2 * RANDOM_STATE,
            )
            ebm.fit(X_train, y_train)
            # store for later use
            with open(model_file, "wb") as file:
                pickle.dump(ebm, file)
        elif dataset_name in [CALIFORNIA_HOUSING, KAGGLE_FLOOD]:
            # regression
            ebm = ExplainableBoostingRegressor(
                interactions=0,
                feature_names=feature_names,
                random_state=3 * RANDOM_STATE,
            )
            ebm.fit(X_train, y_train)
            # store for later use
            with open(model_file, "wb") as file:
                pickle.dump(ebm, file)
        else:
            raise ValueError("Unknown dataset: ", dataset_name)
    return ebm
