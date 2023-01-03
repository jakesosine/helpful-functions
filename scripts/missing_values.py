import pandas as pd 
import numpy as np 
from main import random_state

import seaborn as sns
import matplotlib.pyplot as plt

def get_missing_vals(df:pd.DataFrame,
                     graph:bool=True):
    """
    Descrption: iterate through columns in a dataframe and calculate the 
    proportion of missing values. 
    ------
    Parameters
    ----------
    df: pd.DataFrame
        the initial dataframe
    graph: bool
        True/False value to outline if a graph will be produced
    Returns
    -------
    dict
        a dictionary with colnames as keys and proportion of missing values as
        values
    """
    import pandas as pd
    prop_missing = {}
    for colname in list(df):
        prop_missing[colname] = df[colname].isna().sum()/len(df)
    prop_missing = dict(sorted(prop_missing.items(),key= lambda item:item[1]))
    if graph == True:
        fig, ax = plt.subplots(figsize=(12,8))
        sns.barplot(y=list(prop_missing.keys()),x=list(prop_missing.values()),ax=ax,color='black')
        ax.set_ylabel('Columns', fontsize=20, labelpad=10)
        ax.set_xlabel('Proportion of Missing Values', fontsize=20, labelpad=10)
        ax.set_xlim([0,1])
        ax.tick_params(axis='both',labelsize=20)
        plt.tight_layout()
        plt.savefig('./figures/proportion_missing.pdf')
        plt.show()
    return prop_missing

def iterative_imputer(df,
                      random_state=random_state):
    """
    Description: Missing values are not ideal. There are many ways to deal with 
    missing values. There are simple imputers, like replacing missing values with zero, 
    or the mean. Here is an example of a bit more complex of imputation techniques, using 
    iterative imputation using linear regression. 
    ------
    Parameters
    ----------
    df: pd.DataFrame
        the initial dataframe passed into the function
    random_state: int 
        Random state integer to set the seed for the randomness. Often corresponds
        to years of giants world series victories. 
    Returns
    -------
    df: pd.DataFrame
        dataframe imputed
    imp: IterateiveImputer
        imputer object to use on the test set. 
    """
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    imp = IterativeImputer(estimator=lr,missing_values=np.nan,max_iter=10,verbose=2,imputation_order='roman',random_state=random_state)
    X = imp.fit_transform(df)
    df = pd.DataFrame(X,columns=list(df),index=df.index)
    return df, imp


if __name__ == '__main__':
    test_df = pd.read_csv('./data/01_raw/test.csv')
    prop_missing = get_missing_vals(test_df)
    print(prop_missing)
