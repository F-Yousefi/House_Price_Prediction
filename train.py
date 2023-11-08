# -*- coding: utf-8 -*-
"""Module providing a machine learning model powered by linear regression."""
import os
from optparse import OptionParser
from dataset.config import Config
from dataset.dataset import data_loader, divar_dataset_correction
from linear_regression.model import LinearRegressionGD, LinearRegressionNE
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')


parser = OptionParser()
parser.add_option("-p", "--path", dest="train_path",
                  help="Path to training data.", default=None)
parser.add_option(
    "-m",
    "--method",
    dest="method",
    help="choose between \'normal-equation\' and \'gradient-descent\'\
                  default value is \'normal-equation\' ",
    default="normal-equation")

(options, args) = parser.parse_args()

config = Config()
APPLICATION_NAME = config.application_name
CURRENT_YEAR = config.current_year

if options.train_path is not None:
    if os.path.isfile(options.train_path):
        TRAIN_DATA_PATH = options.train_path
    else:
        raise FileNotFoundError(
            APPLICATION_NAME +
            ": Sorry, .csv file cannot be found in the path \"{0}\"".format(
                options.train_path))
else:
    TRAIN_DATA_PATH = config.train_data_path

METHOD = options.method


ds = data_loader(TRAIN_DATA_PATH)
x, y = divar_dataset_correction(ds)
del ds

x_norm = normalize(x, axis=0, norm="max")
y_norm = normalize(y, axis=0, norm="max")
del x, y

print("\nSelected Method: ", METHOD)

if METHOD == "gradient-descent":
    model = LinearRegressionGD()  # Using the Gradient Descent method
    model = model.fit(x_norm, y_norm, alpha=0.54, acceptable_cost=1e-5)
    MESSAGE = "\nThe best possible score is 1.0 and it can be negative (because the\
      \nmodel can be arbitrarily worse). A constant model that always predicts\
      \nthe expected value of y, disregarding the input features, would get\
      \na R^2 score of 0.0."
    print(MESSAGE)
    print("\nSCORE : ", model.score(x_norm, y_norm))

elif METHOD == "normal-equation":
    model = LinearRegressionNE()  # Using the Gradient Descent method
    model = model.fit(x_norm, y_norm)
    MESSAGE = "\nThe best possible score is 1.0 and it can be negative (because the\
      \nmodel can be arbitrarily worse). A constant model that always predicts\
      \nthe expected value of y, disregarding the input features, would get\
      \na R^2 score of 0.0."
    print(MESSAGE)
    print("\nSCORE : ", model.score(x_norm, y_norm))
