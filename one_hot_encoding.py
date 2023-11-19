import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def oneHotEncode(data):
    '''
    Takes a pandas dataframe.
    Uses sklearn's OneHotEncoder package that encodes categorical integers into a matrix of integers.
    Returns an updated dataframe with one hot encoding. 
    '''
    df = data.infer_objects() 
    enc = OneHotEncoder()

    # limit dataset to only categorical columns
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    catCols = list(set(cols) - set(num_cols))
    # print('categorical columns: ', catCols)

    # transform
    catFeatArray = enc.fit_transform(df[catCols]).toarray()

    # the values for categorical columns
    catFeatLabels = enc.categories_

    # get all values of all categorical cols in one array
    catFeatLabels = np.array(catFeatLabels).ravel()
    # print('categorical column values in one array: ', catFeatLabels)

    # create 1 array of all categorical features with values [0,1]
    catFeats = pd.DataFrame(catFeatArray, columns = catFeatLabels)
    # print('array of features with values: ', catFeats)

    # join new features with original df 
    dfNew = pd.concat([df, catFeats], axis=1)
    # print('new array with categorical columns included: ', dfNew)

    # final dataframe to be only numerical
    dfNew = dfNew.select_dtypes(exclude=['object'])
    # print('final array with categorical columns excluded: ', dfNewn)

    return dfNew


