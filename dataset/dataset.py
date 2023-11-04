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
        return dataset
    except BaseException:
        raise FileNotFoundError(
            APPLICATION_NAME +
            ": Sorry, .csv file cannot be found in the path \"{0}\"".format(path))


def chngtoen(row):
    row["Price"] = int(
        row["Price"].replace(
            " تومان",
            "").replace(
            "٬",
            "").replace(
                " ",
                "").replace(
                    "توافقی",
            "-1"))
    return row


def divar_dataset_correction(divar_dataset):

    divar_dataset = divar_dataset.drop(51)
    divar_dataset = divar_dataset.where(
        divar_dataset["Warehouse"] == False).dropna()
    divar_dataset = divar_dataset.apply(chngtoen, axis=1)
    divar_dataset = divar_dataset.where(divar_dataset["Price"] != -1).dropna()
    divar_dataset["Construction"] = CURRENT_YEAR - \
        divar_dataset["Construction"].astype(int)
    divar_dataset["Elevator"] = divar_dataset["Elevator"].astype(
        bool).astype(int)
    divar_dataset = divar_dataset[[
        "Area", "Construction", "Room", "Price", "Elevator"]]
    divar_dataset = divar_dataset.reset_index(drop=True)

    x = divar_dataset[['Area', 'Construction', 'Room', 'Elevator']].to_numpy()
    y = divar_dataset["Price"].to_numpy()
    y = y.reshape((len(y), 1))

    return x, y
