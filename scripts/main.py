import os 

# tabular data / linear algebra
import pandas as pd 
import numpy as np
# graphing packages 
import seaborn as sns
import matplotlib.pyplot as plt
random_state = 2010







if __name__ == "__main__":
    df = pd.read_csv('./data/01_raw/test.csv')
    print(df.head())
