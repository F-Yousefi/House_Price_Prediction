# -*- coding: utf-8 -*-
"""Module providing a machine learning model powered by linear regression."""
import os
from optparse import OptionParser
from linear_regression.config import Config
from linear_regression.model import LinearRegression
from dataset.dataset import data_loader, divar_dataset_correction
from utils import cost_plotter, scikit_truth_weights


parser = OptionParser()
parser.add_option("-p", "--path", dest="train_path",
                  help="Path to training data.", default=None)
parser.add_option(
    "-s",
    "--save",
    dest="model_save_path",
    help="filname for saving Model. You will need this option to predict.",
    default="LinearRegression")

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

if options.model_save_path is not None:
    SAVE_PATH = options.model_save_path

ds = data_loader(TRAIN_DATA_PATH)
x, y = divar_dataset_correction(ds)

model = LinearRegression()
model = model.fit(x, y)
repo = model.get_report()
x, y = model.get_params()[0:2]
cost_plotter(repo)
scikit_truth_weights(x, y, repo)

if SAVE_PATH is not None:
    model.save(SAVE_PATH)
