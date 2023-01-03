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

def get_variance_inflation_factor(df,
                                  variables:list,
                                  plot:bool=True):
    """
    Description:Variance Inflation Factor (VIF) is a score telling us about how
    well an independent variable is predictable using other independent variables. 
    This may indicate that we need to adjust our feature space in order to reduce
    the likelihood of multicollinearity.  
    ------
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe needs to be less of np.inf or -np.inf values as well as null
        values
    variables: list 
        list of variables to check for multicollinearity
    Returns
    -------
    vif_df:pd.DataFrame
        dataframe with VIF metric
    Other: graph
        Graph that shit
    """              
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_df = pd.DataFrame()
    df = df[variables]
    vif_df['features'] = df[variables].columns
    vif_df['score'] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    vif_df = vif_df.sort_values(by=['score'],ascending=False)
    if plot == True:
        fig, ax = plt.subplots(figsize=(12,8))
        sns.barplot(data=vif_df,x='score',y='features',ax=ax,color='black')
        ax.set_xlabel('VIF Score', fontsize=18,labelpad=15)
        ax.set_ylabel('Features', fontsize=18,labelpad=15)
        plt.show()
    return vif_df  


if __name__ == '__main__':
    print('test')