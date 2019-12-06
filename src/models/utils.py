import numpy as np
import pandas as pd
import cv2

def balance(dataframe, columns, reference, number):
    """ returns balanced dataframe
    """
    new_dataframe = pd.DataFrame({column: [] for column in columns})
    for label in dataframe[reference].value_counts().index:
        new_dataframe = new_dataframe.append(dataframe[dataframe[reference]==label].iloc[:number])
    return(new_dataframe)


if __name__ == "__main__":
    # DEV TEST
    dataframe = pd.read_csv('../csv/cleaned_data.csv')
    columns = ['image', 'emotion']
    number = 100
    print(balance(dataframe, columns, 'emotion', number)['emotion'].value_counts())