# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from linear_regression.config import Config


config = Config()
APPLICATION_NAME = config.application_name
CURRENT_YEAR = config.current_year


def data_loader(path):
    try:
        dataset = pd.read_csv(path)
        print(
            APPLICATION_NAME +
            ": Dataset located in {0} has been loaded.".format(path))
        print(dataset.info())
        return dataset
    except BaseException:
        raise FileNotFoundError(
            APPLICATION_NAME +
            ": Sorry, .csv file cannot be found in the path \"{0}\"".format(path))


def remove_outliers(dataset):
    q1 = dataset.quantile(0.25)
    q3 = dataset.quantile(0.75)
    IQR = q3 - q1
    lower = q1 - 1.5 * IQR
    upper = q3 + 1.5 * IQR

    upper_array = np.where(dataset >= upper)[0]
    lower_array = np.where(dataset <= lower)[0]
    return upper_array, lower_array


def divar_dataset_correction(dataset):

    def is_int(element: any) -> bool:
        if element is None:
            return False
        try:
            int(element)
            return True
        except ValueError:
            return False

    dataset = dataset.dropna()
    boolean_features = ['Parking', 'Warehouse', 'Elevator']
    dataset[boolean_features] = dataset[boolean_features].astype('int64')
    dataset = dataset[dataset["Area"].apply(is_int)]
    dataset["Area"] = dataset["Area"].astype('int64')
    dataset = dataset.drop(columns=['Price(USD)'])
    dataset.drop(columns=["Address"], inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    considered_outliers = dataset[["Area", "Price"]]
    upper_array, lower_array = remove_outliers(considered_outliers)
    print("\n {} outliers have been found and removed!".format(len(upper_array)))
    dataset.drop(index=upper_array, inplace=True)
    dataset = dataset.copy().reset_index(drop=True)
    x = dataset[dataset.columns[:-1]].to_numpy()
    y = dataset[dataset.columns[-1]].to_numpy().reshape((-1, 1))

    return x, y
