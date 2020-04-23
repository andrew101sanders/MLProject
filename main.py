"""
    Machine Learning Project

    Authors:
        Andrew Sanders
        Xavier Hodges

    Purpose:
        To create a program that provides homeowners a recommended AirBnb
        listing price based on the description of their living space.
"""

import requests
import pandas as pd
from PIL import Image
import io
import os

datacsv = pd.read_csv('airbnb-listings.csv', sep=';', low_memory=False)

print(datacsv)
