import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def ordinalEncode(data):
    '''
    Takes a pandas dataframe.
    Uses sklearn's OrdinalEncoder package that encodes categorical values into an array of integers.
    Returns an updated dataframe with one ordinal encoding. 
    '''
    df = data.infer_objects()
    enc = OrdinalEncoder()

    # limit dataset to only categorical columns
    cols = df.columns
    numCols = df.select_dtypes(exclude=['bool', 'object'])
    categoriesDf = df[list(set(cols) - set(numCols))]
    # fit encoder
    enc.fit(categoriesDf)

    # transform categorical columns
    encodedColumns = enc.transform(categoriesDf)

    # Take the encoded columns and fit them to the original shape of the df
    encodedDf = pd.DataFrame(encodedColumns, columns= categoriesDf.columns)

    # Drop the old, unencoded columns
    df = df.drop(columns=categoriesDf.columns)

    # Join the new columns in their place
    df = df.join(encodedDf)
    
    return df
