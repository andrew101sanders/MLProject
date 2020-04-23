"""
    Download the images in the dataset and save them locally

    Authors:
        Andrew Sanders
        Xavier Hodges
"""
import pandas as pd
import requests
import concurrent.futures
from PIL import Image
import io

datacsv = pd.read_csv('airbnb-listings.csv', sep=';', low_memory=False)
data = datacsv.query('`Number of Reviews` > 20').query('`Review Scores Value` > 9.0').query('`Room Type` == "Entire home/apt"')
indexes = data.index.to_list()
final = data['Price'][indexes]

urls = list(data['XL Picture Url'][indexes])


def load_url(url, index):
    """
    Http get request to the given url and return the image
    :param url:
    :param index:
    :return [image, index]:
    """
    if not pd.isna(url):
        ans = requests.get(url, timeout=60)
        return ans.content, index
    return -1, index


def downloadImages():
    """
    Download the images in the dataset
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=5000) as executor:
        future_to_url = (executor.submit(load_url, urls[i], i) for i in range(len(indexes)))
        for future in concurrent.futures.as_completed(future_to_url):
            try:
                data = future.result()
                if data[0] != -1:
                    Image.open(io.BytesIO(data[0])).save('D:/Pictures2/' + str(indexes[data[1]]) + '.jpg')
            except Exception as exc:
                print(exc)
downloadImages()

'''
    Split the data
import os, shutil, numpy as np
source1 = 'D:/PicturesTrain'
dest1 = 'D:/PicturesTest'
files = os.listdir(source1)
for f in files:
       if np.random.rand(1) < 0.2:
           shutil.move(source1 + '/' + f, dest1 + '/' + f)
'''