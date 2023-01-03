import numpy as np 
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt 

def histplot(data,
             x,
             figsize=(12,8),
             savename='histplot'):
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data=data,
                 x=x)
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.set_ylabel('Count',fontsize=26)
    ax.set_xlabel(' '.join(f'{x}'.split('_')).title(),fontsize=26)
    sns.despine(top=True,right=True, offset=5)
    plt.tight_layout()
    plt.savefig(f'./figures/{savename}.pdf')
    plt.show()
    

def scatterplot(data,
                x,
                y, 
                hue=None,
                figsize=(12,8),
                savename='scatterplot'):
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=data,
                    x=x,
                    y=y,
                    hue=hue,
                    alpha=.1,
                    color='black')
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.set_ylabel(' '.join(f'{y}'.split('_')).title(),fontsize=26)
    ax.set_xlabel(' '.join(f'{x}'.split('_')).title(),fontsize=26)
    sns.despine(top=True,right=True, offset=5)
    plt.tight_layout()
    plt.savefig(f'./figures/{savename}.pdf')
    plt.show()   
if __name__ == '__main__':
    df = pd.read_csv('./data/01_raw/train.csv')    
    histplot(df,'launch_speed')
    scatterplot(df, x='launch_angle',y='launch_speed',hue='is_home_run')



