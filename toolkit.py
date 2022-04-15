import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class ObtainData:
    """This class contains methods for obtaining data and loading it."""

    def __init__(self):
        pass

    def fetch_data(self, data_url, data_path, dataName):
        """This function fetches training data and saves it to machine.

        Parameters
        ----------
            data_url : str
                This is the url of the data to fetch
            data_path : str
                This is the path to where the data is saved
            dataName : str
                The name to give to the data after download

        Returns
        -------
            None
        """
        if not os.path.isdir(data_path):
            os.makedirs(data_path)
        tgz_path = os.path.join(data_path, dataName)
        urllib.request.urlretrieve(data_url, tgz_path)
        data_tgz = tarfile.open(tgz_path)
        data_tgz.extractall(path=data_path)
        data_tgz.close()
        print('done')

    def loadData(self, data_path, dataName):
        """This function loads csv data to a pandas DataFrame.

        Parameters
        ----------
            data_path : str
                This is the path to where the data is saved
            dataName : str
                The name to give to the data after download

        Returns
        -------
            `pandas` DataFrame
        """
        csv_path=os.path.join(data_path, dataName)
        return pd.read_csv(csv_path)


class ProcessData:
    """This class contains methods for processing data for machine learning."""

    def __init__(self):
        pass

    def split_train_test(self, data, test_ratio, seed=42):
        """This function splits data into training data and testing data.
        
        Parameters
        ----------
            data : numpy.ndarray or `pandas` DataFrame
                The data that is to be split
            test_ratio : float
                The percentage of the test data
            seed : int
                The value to seed the randomization for repeatability
        Returns
        -------
            tuple
                The return is the training data and the test data
        """
        train_set, test_set = train_test_split(data, test_size=test_ratio, random_state=seed)
        return train_set, test_set