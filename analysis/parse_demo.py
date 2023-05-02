
import pandas as pd
import numpy as np
import re
from sys import argv
import sys, os

sc, basedir = argv

def parse_cols(col_name):
    sex = ['м', 'ж']
    types = ['город', 'село']
    col_list = col_name.split()
    result = pd.Series(np.empty((4)))
    result[:] = np.nan
    result[0] = ''
    for col_item in col_list:
        if col_item in sex:
            result[1] = col_item
        elif col_item in types:
            result[2] = col_item
        elif re.fullmatch('^\d+-\d+$', col_item) or re.fullmatch('^\d+$', col_item):
            result[3] = col_item
        else:
            result[0] = f'{result[0]} {col_item.replace("населения", "")}'.strip()
    result = result.fillna('все') 
    return result

def fill_dataframes(line_prev, line):
    global demo_df, new_demo_df, f_df, m_df
    if line[0] == 'F':
        f_df.loc[(f_df['Возраст'] == line[3])&(f_df['Тип'] == line[2]), line[0]] = demo_df[line_prev]
    elif line[0] == 'm':      
        m_df.loc[(m_df['Пол'] == line[1])&(m_df['Возраст'] == line[3])&(m_df['Тип'] == line[2]), line[0]] = demo_df[line_prev]       
    else:
        if line[3] == 'все':  
            new_demo_df.loc[(new_demo_df['Пол'] == line[1])&(new_demo_df['Тип'] == line[2]), line[0]] = demo_df[line_prev]    



demo_df = pd.read_csv(os.path.join(basedir,'./analysis/data/demographics.csv'), sep=';', encoding='cp1251', decimal=',')
parsed_cols = pd.Series(demo_df.columns).apply(lambda x: parse_cols(x))


parsed_demo_cols = list(parsed_cols[0].unique())
parsed_demo_cols.remove('F')
parsed_demo_cols.remove('m')
new_demo_df = demo_df[['Год','Субъект']]
new_demo_df = new_demo_df.merge(pd.DataFrame({'Пол':['м', 'ж', 'все']}), how='cross').merge(pd.DataFrame({'Тип':['город', 'село', 'все']}), how='cross')
new_demo_df[parsed_demo_cols[2:]] = np.nan
new_demo_df = new_demo_df.set_index(['Год', 'Субъект'])

f_df = demo_df[['Год','Субъект']]
f_df = f_df.merge(pd.DataFrame({'Тип':['город', 'село', 'все']}), how='cross').merge(pd.DataFrame({'Возраст':list(parsed_cols[3].unique())}), how='cross')
f_df[['F']] = np.nan
f_df = f_df.set_index(['Год', 'Субъект'])

m_df = demo_df[['Год','Субъект']]
m_df = m_df.merge(pd.DataFrame({'Пол':['м', 'ж', 'все']}), how='cross').merge(pd.DataFrame({'Тип':['город', 'село', 'все']}), how='cross').merge(pd.DataFrame({'Возраст':list(parsed_cols[3].unique())}), how='cross')
m_df[['m']] = np.nan
m_df = m_df.set_index(['Год', 'Субъект'])
demo_df = demo_df.set_index(['Год', 'Субъект'])

n = pd.Series(demo_df.columns).apply(lambda x: fill_dataframes(x, parse_cols(x)))

demo_df = demo_df.reset_index() 
f_df = f_df.reset_index()
m_df = m_df.reset_index()
new_demo_df = new_demo_df.reset_index()

m_df = m_df.drop(index=m_df[m_df['m'].isna()].index).reset_index(drop=True)
f_df = f_df.drop(index=f_df[f_df['F'].isna()].index).reset_index(drop=True)
new_demo_df = new_demo_df.drop(index=new_demo_df[new_demo_df.isna().sum(1) == (len(new_demo_df.columns)-4)].index).reset_index(drop=True)


new_demo_df.to_csv(os.path.join(basedir,'./analysis/data/new_demo_df.csv'), sep=';', encoding='cp1251', index=False, decimal=',')
f_df.to_csv(os.path.join(basedir,'./analysis/data/f_df.csv'), sep=';', encoding='cp1251', index=False, decimal=',')
m_df.to_csv(os.path.join(basedir,'./analysis/data/m_df.csv'), sep=';', encoding='cp1251', index=False, decimal=',')