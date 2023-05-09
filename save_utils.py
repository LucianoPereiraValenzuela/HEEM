import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union
from utils import _question
from datetime import datetime


def _timestamp():
    return datetime.now().strftime("%H:%M:%S, %d/%m/%Y")


def _save_figure(fig: plt.Figure, file_dic: str, dpi: Optional[int] = 600):
    # Save the figure with the corresponding file direction and the correct extension
    fig.savefig(file_dic, bbox_inches="tight", dpi=dpi)


def _save_data(data: dict, file_dic: str):
    if type(data) == dict:
        if 'timestamp' not in data.keys():
            data['timestamp'] = []
        data['timestamp'].append(_timestamp())

    np.save(file_dic, data)


def save_object(object_save: Union[dict, plt.Figure], name: str, overwrite: Optional[bool] = None,
                extension: Optional[str] = None, dic: Optional[str] = None, ask: Optional[bool] = True):
    """
    Save a given figure or data encoded in a dictionary. Introduce the name with which we want to save the file. If the
    file already exist, then we will be asked if we want to overwrite it.
    We can also change the extension used for the save file.

    Parameters
    ----------
    object_save: plt.Figure or dict
        Matplotlib figure or dictionary with the data to save
    name: str
        Name of the file in which save the data
    overwrite: bool (optional, default=None)
        Condition to overwrite or not the data. If a value is not given the function will ask by default
    extension: str (optional, default=None)
        Extension of the save file. By default, extension='npy' for data, and extension='pdf' for figures
    dic: str (optional, default=None)
        Directory to save the data. By default, dic='data/
    ask: bool (optional, default=True)
        If True, the question to overwrite is printed.
    """

    # Check the type of the data
    if type(object_save) is plt.Figure:
        object_type = 'figure'
        save_function = _save_figure
    else:
        object_type = 'data'
        save_function = _save_data

    if dic is None:
        dic = 'data/'

    # Search the file in a total of 10 parent folders
    if dic != '':
        counter_max = 10
        back = 0
        while True:
            if back < counter_max:
                if os.path.exists('../' * back + dic):
                    break
                else:
                    back += 1
            else:
                raise Exception('Folder not found')

        dic = '../' * back + dic

    if extension is None:
        if object_type == 'figure':
            extension = 'pdf'
        else:
            extension = 'npy'

    file_dic = dic + name

    if overwrite is None:  # If the user does not give a preference for the overwriting
        if os.path.isfile(file_dic + '.' + extension):  # If the file exists in the folder
            if ask:  # The function will ask if the user want to overwrite the file
                overwrite = _question(name=file_dic + '.' + extension)
            else:
                overwrite = False
        else:
            overwrite = True  # If the file does not exist, them the object will be saved

    if overwrite:
        save_function(object_save, file_dic + '.' + extension)
        print(object_type, 'saved at:', file_dic + '.' + extension)
