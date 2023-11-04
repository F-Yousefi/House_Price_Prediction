"""Module providing a function to predict the price of a house based on its features."""
from optparse import OptionParser
import pickle
import locale
import numpy as np



parser = OptionParser()
parser.add_option(
    "-l",
    "--load",
    dest="model_load_path",
    help="filname for loading Model. You will need this option to predict.",
    default="LinearRegression")
(options, args) = parser.parse_args()

SAVE_PATH = options.model_load_path


with open('{0}.pickle'.format(SAVE_PATH), 'rb') as handle:
    model = pickle.load(handle)
    print("Model Linear Regression has been loaded.")

print("The model is ready to use:\n Please enter the properties of the house.\n")
x1 = input("Please enter area of the house in m^2:")
x2 = input("Please enter how old the house is:")
x3 = input("Please enter how many rooms it has:")
x4 = input("Please enter 1 if the elevator is provided, otherwise enter 0 :")
predicted_price = model.predict(np.array([x1, x2, x3, x4]).astype(float))
predicted_price /= 1.0e+6
predicted_price = int(predicted_price) * 1000000
locale.setlocale(locale.LC_ALL, '')
print("Price of the house is estimated:", str(f'{predicted_price:n}'), "TOMAN")
