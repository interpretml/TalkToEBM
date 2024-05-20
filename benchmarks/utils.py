""" Load datasets and models for testing purposes.
"""

import pandas as pd
import os
import openai
import guidance
import pickle

from sklearn.model_selection import train_test_split

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)

RANDOM_STATE = 1498672

IRIS_SETOSA = (
    "Iris-setosa"  # binary classification of Iris-setosa vs. all other species
)
TITANIC = "Titanic"
SPACESHIP_TITANIC = "Spaceship-Titanic"
CALIFORNIA_HOUSING = "California-Housing"
OPENML_DIABETES = "OpenML-Diabetes"

DATASETS = [
    CALIFORNIA_HOUSING,
    OPENML_DIABETES,
    IRIS_SETOSA,
    TITANIC,
    SPACESHIP_TITANIC,
]


def openai_setup_gpt3_5():
    openai.organization = os.environ["OPENAI_API_ORG"]
    openai.api_key = os.environ["OPENAI_API_KEY"]
    return guidance.llms.OpenAI("gpt-3.5-turbo-0125")


def openai_setup_gpt4():
    openai.organization = os.environ["OPENAI_API_ORG"]
    openai.api_key = os.environ["OPENAI_API_KEY"]
    return guidance.llms.OpenAI("gpt-4o-2024-05-13")


def get_avaialble_datasets():
    return DATASETS


def get_dataset_description(dataset_name):
    """Returns: dataset description (str), dataset y axis description (str)"""
    pass


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
        df = df.drop(columns=["PassengerId", "Name", "Survived"])
        X_data = df.values
        feature_names = df.columns.tolist()
    elif dataset_name == IRIS_SETOSA:
        df = pd.read_csv("../data/IRIS.csv")
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
    else:
        raise ValueError("Unknown dataset: ", dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, feature_names


def get_ebm(dataset_name):
    """
    Returns: the ebm
    """
    X_train, X_test, y_train, y_test, feature_names = get_dataset(dataset_name)
    model_file = f"../models/{dataset_name}"
    if os.path.isfile(model_file):  # check if model file exists
        # load with pickle
        with open(model_file, "rb") as file:
            ebm = pickle.load(file)
    else:  # otherwise train and save
        if dataset_name in [SPACESHIP_TITANIC, IRIS_SETOSA, TITANIC, OPENML_DIABETES]:
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
        elif dataset_name in [CALIFORNIA_HOUSING]:
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
