import os
import pandas as pd
from glob import glob
from tqdm import tqdm
# https://www.kaggle.com/haithemhermessi/sanad-dataset
# This python code loads the SANAD dataset in tabular form

MAIN_PATH = 'SANAD'

txt = list()
label = list()


for path in tqdm(glob('SANAD/*')):
    for file in tqdm(glob(f'{path}/*.txt')):
        txt.append(open(file, 'r').read())
        label.append(path.replace('/Users/mohammadkhaled/Desktop/arabert/SANAD/', ''))
        
df = pd.DataFrame.from_dict({'Post Body':txt, 'Topic Category':label})
df.to_csv('SANAD.csv', index=False)

