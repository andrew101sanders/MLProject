"""
    Machine Learning Project

    Authors:
        Andrew Sanders
        Xavier Hodges

    Purpose:
        To create a program that provides homeowners a recommended AirBnb
        listing price based on the description of their living space.
"""

import pandas as pd

datacsv = pd.read_csv('airbnb-listings.csv', sep=';', low_memory=False)

print(datacsv['Price'].mean())