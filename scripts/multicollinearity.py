import pandas as pd 
import numpy as np 


def get_collinearity(df:pd.DataFrame,
                     low:float=-.65,
                     high: float=.65,) -> pd.DataFrame:
    """
    Description: get collinearity of the feature space of a dataframe.
    Function takes in a dataframe and returns a styled dataframe. 
    Excludes categorical and object datatypes from the styling. 
    ------
    Parameters
    ----------
    df: pd.DataFrame
        the initial dataframe passed into the function

    Returns
    -------
    corr_df: pd.DataFrame
        dataframe that has correlation between different features.  
    """              
    corr_df = (df
          .select_dtypes(exclude=['category','object'])
          .corr()
          .style
          .background_gradient(low=low,
                               high=high))
    
    corr_df_for_dict = ((df.corr()>=high).unstack()
                                         .reset_index(drop=False)
                                         .rename(columns={'level_0':'feature_1',
                                                          'level_1':'feature_2',
                                                          0:'boolean_mask'})
                                         .sort_values(by='feature_1'))
    
    corr_df_for_dict['boolean_mask'] = corr_df_for_dict['boolean_mask'] *1   
    corr_df_for_dict = corr_df_for_dict.loc[(corr_df_for_dict['boolean_mask']==1) & (corr_df_for_dict['feature_1']!=corr_df_for_dict['feature_2']),:].reset_index()
    stupid_ml_dct = {}
    for i in range(len(corr_df_for_dict)):
        if corr_df_for_dict['feature_1'][i] in stupid_ml_dct.keys() or (corr_df_for_dict['feature_1'][i] in stupid_ml_dct.values()):
            continue 
        else: 
            stupid_ml_dct[corr_df_for_dict['feature_1'][i]] = corr_df_for_dict['feature_2'][i]
    return corr_df, stupid_ml_dct


if __name__ == '__main__':
    print('test')