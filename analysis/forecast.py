import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import itertools
from pathlib import Path
from keras import models
from sys import argv
import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(42)

script, file_path, base_year, t = argv


# ### Список субъектов
regions = pd.read_csv("./data/Территории.csv", sep=";")
region_encoder = LabelEncoder()
regions['Код'] = region_encoder.fit_transform(regions['Наименование'])
regions.iloc[:, 1:-1] = regions.iloc[:, 1:-1].fillna('').apply(lambda x: x.apply(lambda x: region_encoder.transform([s.strip() for s in x.split(',') if s])))
regions = regions.set_index('Код')
for col in regions.columns[1:]:
    for i in regions.index:
        regions.loc[regions[col][i], col.replace('Состав', 'Часть')] = i
regions[regions.columns[-4:]] = regions[regions.columns[-4:]].fillna(-1).astype(np.int64)


# ## Прогнозирование
if not file_path:
    file_path = "./data/demographics.csv"
if not base_year:
    base_year = 2021
if not t:
    t = 10
base_year = int(base_year)
t = int(t)
file_path = Path(file_path)



forecast_df = pd.DataFrame(columns=['Год', 'Субъект', 'Период'])
for pair in list(itertools.product(*[range(base_year+1, base_year+t+1), regions.index])):
    forecast_df = pd.concat([forecast_df, pd.DataFrame([[*pair, pair[0]-base_year]], columns=forecast_df.columns)], ignore_index=True)
forecast_df = forecast_df.astype(np.int64)

if Path("./data/forecast_df.csv").is_file():
    forecast_df_real = pd.read_csv("./data/forecast_df.csv", sep=';', encoding='cp1251', decimal=',')
    forecast_df = forecast_df.set_index(['Год', 'Субъект','Период'])
    forecast_df = forecast_df.loc[list(set(forecast_df.index) - set(forecast_df_real.set_index(['Год', 'Субъект','Период']).index))].reset_index().sort_values(by='Год')

if forecast_df.shape[0] == 0:
    sys.exit()


demographics = pd.read_csv(file_path, sep=";", decimal=',', encoding='cp1251')


# #### Метод экстраполяции по среднему абсолютному приросту
forecast_df = forecast_df.merge(demographics[demographics['Год'] == base_year].set_index('Субъект')['Численность населения'], on='Субъект')\
                         .merge(demographics.groupby(['Субъект']).mean()['Общий прирост'], on='Субъект').rename(columns={'Численность населения': 'Базовая численность'})
forecast_df['Экстраполяция по приросту'] = (forecast_df['Базовая численность']+forecast_df['Общий прирост']*forecast_df['Период']).round()
forecast_df = forecast_df.drop(columns=['Общий прирост'])


# #### Метод экстраполяции по среднему темпу роста
forecast_df = forecast_df.merge(demographics.groupby(['Субъект']).mean()['Темп прироста численности населения']/100+1, on='Субъект').rename(columns={'Темп прироста численности населения': 'Средний рост'})
forecast_df['Экстраполяция по темпу роста'] = (forecast_df['Базовая численность']*forecast_df['Средний рост'].pow(forecast_df['Период'])).round()
forecast_df = forecast_df.drop(columns=['Средний рост'])


# #### Метод экстраполирования по экспоненте
forecast_df = forecast_df.merge(demographics.groupby(['Субъект']).mean()['Темп прироста численности населения']/100, on='Субъект')
forecast_df['Экстраполяция по экспоненте'] = (forecast_df['Базовая численность']*np.exp(1)**(forecast_df['Темп прироста численности населения']*forecast_df['Период'])).round()
forecast_df = forecast_df.drop(columns=['Темп прироста численности населения'])


# #### Метод передвижки возрастов (метод компонент)
death_tables = pd.read_csv("./data/Краткие таблицы смертности.csv", sep=";", decimal='.', encoding='cp1251')
age_structure = pd.read_csv("./data/age_structure.csv", sep=";", decimal=',', encoding='cp1251')


def count_born(death_tables, demographics, year1, year2):
    count_df = death_tables[(death_tables['Год'] == year1)&(death_tables['Тип'] == 'все население')&(death_tables['Пол'] == 'Женщины')&(death_tables['Возраст'].between(5, 50))][['Субъект', 'Возраст', 'Число доживших']]
    count_df['Возраст следующий'] = count_df['Возраст']+5
    count_df = count_df.merge(count_df[['Субъект', 'Возраст', 'Число доживших']], left_on = ['Субъект', 'Возраст следующий'], right_on = ['Субъект','Возраст']).drop(columns = 'Возраст_y').rename(columns={'Возраст_x':'Возраст'})
    count_df['Коэффициент дожития'] = count_df['Число доживших_y'] / count_df['Число доживших_x']

    middle_df = pd.DataFrame(columns=['Субъект', 'Возраст', 'Численность ж'], dtype='int64')
    for i in range(10, 50, 5):
        middle_middle_df = age_structure[(age_structure['Год'] == year2)&(age_structure['Тип'] == 'все население')&(age_structure['Возраст'].between(i, i+4))][['Субъект', 'Численность ж']]
        middle_middle_df['Возраст'] = i
        middle_df = pd.concat([middle_df, middle_middle_df])
    count_df = count_df.merge(middle_df, on = ['Субъект','Возраст'])

    count_df['Численность ж следующий'] = (count_df['Коэффициент дожития']*count_df['Численность ж']).round()
    count_df = count_df.merge(count_df[['Субъект', 'Возраст следующий', 'Численность ж следующий']], left_on = ['Субъект', 'Возраст'], right_on = ['Субъект','Возраст следующий']).drop(columns = ['Численность ж следующий_x', 'Возраст следующий_y']).rename(columns={'Численность ж следующий_y':'Численность ж следующий'})

    middle_df = pd.DataFrame(columns=['Субъект', 'Возраст', 'F'], dtype='int64')
    for i in range(15, 50, 5):
        middle_middle_df = demographics[(demographics['Год'] == year1)][['Субъект', f'F {i}-{i+4}']].rename(columns={f'F {i}-{i+4}':'F'})
        middle_middle_df['Возраст'] = i
        middle_df = pd.concat([middle_df, middle_middle_df])
    middle_df['F'] /= 1000
    count_df = count_df.merge(middle_df, on = ['Субъект','Возраст'])
    count_df['Численность родившихся'] = (count_df['F']*count_df['Численность ж следующий']).round()
    return count_df.groupby('Субъект').sum()['Численность родившихся']


def count_forecast(death_tables, age_structure, year1, year2, born, sex='Оба пола'):
    count_df = death_tables[(death_tables['Год'] == year1)&(death_tables['Тип'] == 'все население')&(death_tables['Пол'] == sex)][['Субъект', 'Возраст', 'Число доживших']]
    count_df['Возраст следующий'] = count_df['Возраст'].apply(lambda x: x+1 if x == 0 else x + 4 if x == 1 else x + 5)
    count_df = count_df.merge(count_df[['Субъект', 'Возраст', 'Число доживших']], left_on = ['Субъект', 'Возраст следующий'], right_on = ['Субъект','Возраст'], how='left').drop(columns = 'Возраст_y').rename(columns={'Возраст_x':'Возраст'})
    count_df['Коэффициент дожития'] = count_df['Число доживших_y'] / count_df['Число доживших_x']
    count_df = count_df.fillna(0)

    col = 'Численность населения'
    if sex == 'Женщины':
        col = col.replace('населения', 'ж')
    elif sex == 'Мужчины':
        col = col.replace('населения', 'м')
    middle_df = pd.DataFrame(columns=['Субъект', 'Возраст', col], dtype='int64')
    middle_middle_df = age_structure[(age_structure['Год'] == year2)&(age_structure['Тип'] == 'все население')&(age_structure['Возраст'] == 0)].set_index('Субъект')[col].reset_index().rename(columns={0:col})
    middle_middle_df['Возраст'] = 0 
    middle_df = pd.concat([middle_df, middle_middle_df])
    middle_middle_df = age_structure[(age_structure['Год'] == year2)&(age_structure['Тип'] == 'все население')&(age_structure['Возраст'].between(1, 4))].groupby('Субъект').sum()[col].reset_index().rename(columns={0:col})
    middle_middle_df['Возраст'] = 1
    middle_df = pd.concat([middle_df, middle_middle_df])

    for i in range(5, 85, 5):
        middle_middle_df = age_structure[(age_structure['Год'] == year2)&(age_structure['Тип'] == 'все население')&(age_structure['Возраст'].between(i, i+4))].groupby('Субъект').sum()[col].reset_index()
        middle_middle_df['Возраст'] = i
        middle_df = pd.concat([middle_df, middle_middle_df])
    middle_middle_df = age_structure[(age_structure['Год'] == year2)&(age_structure['Тип'] == 'все население')&(age_structure['Возраст'].between(85, 105))].groupby('Субъект').sum()[col].reset_index()
    middle_middle_df['Возраст'] = 85    
    middle_df = pd.concat([middle_df, middle_middle_df])
    count_df = count_df.merge(middle_df, on = ['Субъект','Возраст'])

    count_df['Численность следующий'] = (count_df['Коэффициент дожития']*count_df[col]).round()
    count_df = count_df.merge(count_df[['Субъект', 'Возраст следующий', 'Численность следующий']], left_on = ['Субъект', 'Возраст'], right_on = ['Субъект','Возраст следующий'], how='left').drop(columns = ['Численность следующий_x', 'Возраст следующий_y']).rename(columns={'Численность следующий_y':'Численность следующий'})

    count_df = count_df.set_index('Субъект')
    count_df.loc[(count_df['Возраст'] == 0), 'Численность следующий'] = born
    
    return count_df[['Возраст', 'Численность следующий']].rename(columns={'Численность следующий':col}).reset_index()


for year in forecast_df['Год'].unique():
    y = year - 5
    while y > base_year:
        y -= 5
    born = count_born(death_tables, demographics, y, year - 5)
    count_df = count_forecast(death_tables, age_structure, y, year - 5, born, sex='Оба пола')
    count_df['Тип'] = 'все население'
    count_df['Год'] = year
    count_df = count_df.set_index(['Возраст', 'Субъект'])
    
    age_structure = age_structure.set_index(['Возраст', 'Субъект'])
    ind = list(set(count_df.index)-set(age_structure[(age_structure['Год'] == year)&(age_structure['Тип'] =='все население')].index))
    if len(ind) > 0:
        age_structure = pd.concat([age_structure, count_df.loc[ind]])
        age_structure = age_structure.reset_index()
        count_df = count_forecast(death_tables, age_structure, y, year - 5, (born*0.488).round(), sex='Женщины')
        count_df = count_df.set_index(['Возраст', 'Субъект'])
        age_structure = age_structure.set_index(['Возраст', 'Субъект'])
        age_structure.loc[(age_structure['Год'] == year)&(age_structure['Тип'] =='все население'), 'Численность ж'] = count_df['Численность ж']
        age_structure = age_structure.reset_index()
        age_structure['Численность м'] = age_structure['Численность населения'] - age_structure['Численность ж']
    else:
        age_structure = age_structure.reset_index()

age_structure.to_csv('./data/age_structure.csv', sep=';', encoding='cp1251', index=False, decimal=',')


count_df = age_structure[(age_structure['Тип'] =='все население')].groupby(['Субъект', 'Год']).sum()['Численность населения'].reset_index().rename(columns={'Численность населения':'Передвижки'})
forecast_df = forecast_df.merge(count_df, on=['Год', 'Субъект'], how='left')
forecast_df['Передвижки'] = (forecast_df.set_index(['Субъект', 'Год'])['Передвижки']+demographics[demographics['Год'] == base_year].set_index(['Субъект'])['Миграционное сальдо']).reset_index()[0]

age_structure['Субъект'] = region_encoder.inverse_transform(age_structure['Субъект'])
age_structure.to_csv('./data/age_structure_names.csv', sep=';', encoding='cp1251', index=False, decimal=',')

# #### Выравнивание по прямой
def count_t(middle, even, year):
    dif = int(year-middle)
    if not even:
        return dif
    if dif >= 0:
        return list(range(1, 2*abs(dif)+2, 2))[abs(dif)]
    else:
        return -list(range(1, 2*abs(dif), 2))[abs(dif)-1]


count_df = demographics[['Год', 'Субъект', 'Численность населения']]
count_df = count_df.merge(count_df.groupby('Субъект').mean()['Год'].reset_index().rename(columns={'Год':'Середина периода'}).round(), on='Субъект')
count_df = count_df.merge((count_df.groupby('Субъект').count()['Год']%2 == 0).reset_index().rename(columns={'Год':'Четное количество уровней'}), on='Субъект')
count_df['t'] = count_df.apply(lambda x: count_t(x['Середина периода'], x['Четное количество уровней'], x['Год']), axis=1)
count_df['t2'] = count_df['t']**2
count_df['ty'] = count_df['t']*count_df['Численность населения']
a0 = count_df.groupby('Субъект').sum()['Численность населения']/count_df.groupby('Субъект').count()['Год']
a1 = count_df.groupby('Субъект').sum()['ty']/count_df.groupby('Субъект').sum()['t2']
middle_year = count_df.groupby('Субъект').max()['Середина периода']


forecast_df = forecast_df.merge(a0.reset_index().rename(columns={0:'a0'}), on='Субъект').merge(a1.reset_index().rename(columns={0:'a1'}), on='Субъект').merge(middle_year.reset_index().rename(columns={0:'Середина периода'}), on='Субъект')
forecast_df['Выравнивание по прямой'] = (forecast_df['a0'] + forecast_df['a1']*(forecast_df['Год']-forecast_df['Середина периода'])).round()
forecast_df = forecast_df.drop(columns=['a0', 'a1', 'Середина периода'])


# #### Кривая роста Гомперца
count_df = demographics[['Год', 'Субъект', 'Численность населения']].copy()
count_df['Численность населения лог'] = np.log(count_df['Численность населения'])
count_df = count_df.merge(count_df.groupby('Субъект').count()['Год'].reset_index().rename(columns={'Год':'Длина ряда'}), on='Субъект')
count_df = count_df.merge(count_df.groupby('Субъект').min()['Год'].reset_index().rename(columns={'Год':'Начальный год'}), on='Субъект')
count_df['n'] = (count_df['Длина ряда']/3).round()
count_df['Часть'] = (count_df['Год']-count_df['Начальный год'])//count_df['n']

count_count_df = count_df.groupby(['Субъект', 'Часть']).sum()['Численность населения лог'].reset_index().rename(columns={'Численность населения лог':'S'}).pivot(index='Субъект', columns='Часть', values='S').rename(columns={0.0:'S1', 1.0:'S2',2.0:'S3'}).reset_index()
count_count_df = count_count_df.merge(count_df.groupby(['Субъект']).max()[['n', 'Длина ряда']], on='Субъект')
count_count_df['d1'] = count_count_df['S2'] -count_count_df['S1']
count_count_df['d2'] = count_count_df['S3'] -count_count_df['S2']
count_count_df['cn'] = (count_count_df['d2'] / count_count_df['d1']).clip(lower=0, upper=1)
count_count_df['c'] = count_count_df['cn'] ** (1/count_count_df['n'])
count_count_df['lgb'] = count_count_df['d1'] * (count_count_df['c']-1) / (count_count_df['cn'] - 1-1e-1)**2
count_count_df['lga'] = (count_count_df['S1'] - count_count_df['d1'] / (count_count_df['cn'] - 1-1e-1))/count_count_df['n']


forecast_df = forecast_df.merge(count_count_df[['Субъект', 'c',	'lgb', 'lga', 'Длина ряда']], on='Субъект')
forecast_df['Кривая роста Гомперца'] = (np.exp(1)**(forecast_df['lga'] + forecast_df['lgb']*forecast_df['c']**(forecast_df['Период']+forecast_df['Длина ряда']))).round()
forecast_df = forecast_df.drop(columns=['c', 'lgb', 'lga', 'Длина ряда'])


# #### LSTM
use_cols = ['Год', 'Субъект', 'Численность населения', 'Число прибывших', 'Число выбывших', 'Число умерших м село', 'Число умерших м город', 'Число умерших ж село',
            'Число умерших ж город', 'Число умерших младенцев м село', 'Число умерших младенцев м город', 'Число умерших младенцев ж село',
            'Число умерших младенцев ж город', 'Число родившихся м село', 'Число родившихся м город', 'Число родившихся ж село', 'Число родившихся ж город',
            'Число разводов село', 'Число разводов город', 'Число браков село', 'Число браков город', 'Число абортов',
            'Численность малоимуших', 'Численность м село 95-99', 'Численность м село 90-94', 'Численность м село 85-89', 'Численность м село 80-84',
            'Численность м село 75-79', 'Численность м село 70-74', 'Численность м село 65-69', 'Численность м село 60-64', 'Численность м село 5-9',
            'Численность м село 55-59', 'Численность м село 50-54', 'Численность м село 45-49', 'Численность м село 40-44', 'Численность м село 35-39',
            'Численность м село 30-34','Численность м село 25-29', 'Численность м село 20-24', 'Численность м село 15-19', 'Численность м село 10-14',
            'Численность м село 100-104', 'Численность м село 0-4', 'Численность м город 95-99', 'Численность м город 90-94', 'Численность м город 85-89',
            'Численность м город 80-84', 'Численность м город 75-79', 'Численность м город 70-74', 'Численность м город 65-69', 'Численность м город 60-64',
            'Численность м город 5-9', 'Численность м город 55-59', 'Численность м город 50-54', 'Численность м город 45-49', 'Численность м город 40-44',
            'Численность м город 35-39', 'Численность м город 30-34', 'Численность м город 25-29', 'Численность м город 20-24', 'Численность м город 15-19',
            'Численность м город 10-14', 'Численность м город 100-104', 'Численность м город 0-4', 'Численность ж село 95-99', 'Численность ж село 90-94', 
            'Численность ж село 85-89', 'Численность ж село 80-84', 'Численность ж село 75-79', 'Численность ж село 70-74', 'Численность ж село 65-69', 
            'Численность ж село 60-64', 'Численность ж село 5-9', 'Численность ж село 55-59', 'Численность ж село 50-54', 'Численность ж село 45-49', 
            'Численность ж село 40-44', 'Численность ж село 35-39', 'Численность ж село 30-34', 'Численность ж село 25-29', 'Численность ж село 20-24', 
            'Численность ж село 15-19', 'Численность ж село 10-14', 'Численность ж село 100-104', 'Численность ж село 0-4', 'Численность ж город 95-99', 
            'Численность ж город 90-94', 'Численность ж город 85-89', 'Численность ж город 80-84', 'Численность ж город 75-79', 'Численность ж город 70-74', 
            'Численность ж город 65-69', 'Численность ж город 60-64', 'Численность ж город 5-9', 'Численность ж город 55-59', 'Численность ж город 50-54', 
            'Численность ж город 45-49', 'Численность ж город 40-44', 'Численность ж город 35-39', 'Численность ж город 30-34', 'Численность ж город 25-29', 
            'Численность ж город 20-24', 'Численность ж город 15-19', 'Численность ж город 10-14', 'Численность ж город 100-104', 'Численность ж город 0-4']

if Path('./data/demographics_forecast.csv').is_file():
    demographics_forecast = pd.read_csv('./data/demographics_forecast.csv', sep=";", decimal=',', encoding='cp1251')
    working_df = demographics_forecast[demographics_forecast['Базовый год'] == base_year][use_cols+['Базовый год']].reset_index(drop=True)
    if working_df.shape[0] == 0:
        working_df = demographics[demographics['Год'] <= base_year][use_cols].reset_index(drop=True)
        working_df['Базовый год'] = base_year
else:
    demographics_forecast = pd.DataFrame(columns=use_cols+['Базовый год'])
    working_df = demographics[demographics['Год'] <= base_year][use_cols].reset_index(drop=True)
    working_df['Базовый год'] = base_year


scalers = {}
n_features = len(use_cols)-2
n_steps_in, n_steps_out = 5, 1

model_path = Path(f"./keras_models/LSTM_model_{base_year}.keras")

if model_path.is_file():
  LSTM_model = models.load_model(model_path)
  for reg in working_df['Субъект'].unique():
    scalers[reg] = MinMaxScaler()
    scalers[reg].fit_transform(working_df[(working_df['Субъект'] == reg)&(working_df['Год'] <= base_year)].sort_values(by='Год').drop(columns=['Год', 'Субъект', 'Базовый год']))
    for t_period in range(t):
        if t_period not in (working_df[working_df['Субъект'] == reg]['Год'] - base_year-1).values:
            scaled_x = scalers[reg].transform(working_df[working_df['Субъект'] == reg][use_cols[2:]])[-n_steps_in:]
            if scaled_x.flatten().shape[0]%(n_steps_in* n_features) == 0:
                x_pred_line = scaled_x.reshape((1, n_steps_in, n_features))
                y_pred_line = LSTM_model.predict(x_pred_line, verbose=0)
                y_pred = scalers[reg].inverse_transform(y_pred_line[0])
                y_pred = np.clip(y_pred, 0, None).round()
                y_pred = np.append([base_year+t_period+1, reg], y_pred)
                working_df = pd.concat([working_df, pd.DataFrame([y_pred], columns=use_cols)], axis=0, ignore_index=True)

working_df['Базовый год'] = base_year
demographics_forecast = demographics_forecast.drop(index=demographics_forecast[demographics_forecast['Базовый год'] == base_year].index)
demographics_forecast = pd.concat([demographics_forecast, working_df], ignore_index=True)
demographics_forecast.to_csv('./data/demographics_forecast.csv', sep=';', encoding='cp1251', index=False, decimal=',')

forecast_df = forecast_df.merge(working_df[['Субъект', 'Год', 'Численность населения']], on=['Субъект', 'Год'], how='left').rename(columns={'Численность населения': 'LSTM'})


# #### RBFN
class RBFN(nn.Module):
    def __init__(self, centers, n_out=10):
        super(RBFN, self).__init__()
        self.n_out = n_out
        self.n_in = centers.size(1)
        self.num_centers = centers.size(0)

        self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(torch.ones(1,self.num_centers), requires_grad = True)
        self.linear = nn.Linear(self.num_centers + self.n_in, self.n_out, bias=True)
        self.initialize_weights()

    def kernel_fun(self, batches):
        n_input = batches.size(0) 
        A = self.centers.view(self.num_centers,-1).repeat(n_input,1,1)
        B = batches.view(n_input,-1).unsqueeze(1).repeat(1,self.num_centers,1)
        C = torch.exp(-self.beta.mul((A-B).pow(2).sum(2,keepdim=False) ) )
        return C

    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(torch.cat([batches, radial_val], dim=1))
        return class_score
    
    def initialize_weights(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
    
    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)
        
class RBFN_TS(object):
    def __init__(self, args):
        self.max_epoch = args.epoch
        self.trainset = args.dataset[0]
        if args.testset:
          self.testset = args.dataset[1]
        else:
          self.testset = args.dataset[0]
        self.model_name = args.model_name
        self.lr = args.lr
        self.n_in = args.n_in
        self.n_out = args.n_out
        self.num_centers = args.num_centers
        self.centers = torch.rand(self.num_centers,self.n_in)

        self.model = RBFN(self.centers, n_out=self.n_out)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fun = nn.MSELoss()

    def train(self, epoch=1):
        self.model.train()
        for epoch in range(min(epoch,self.max_epoch)):
            X = torch.from_numpy(self.trainset[0]).float()
            Y = torch.from_numpy(self.trainset[1]).float()        

            self.optimizer.zero_grad()             
            Y_prediction = self.model(X)         
            cost = self.loss_fun(Y_prediction, Y) 
            cost.backward()                   
            self.optimizer.step()                  

            print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, cost.item()))
        print(" [*] Training finished!")

    def test(self):
        self.model.eval()
        X = torch.from_numpy(self.testset[0]).float()
        Y = torch.from_numpy(self.testset[1]).float()        

        with torch.no_grad():             
            Y_prediction = self.model(X)         
            cost = self.loss_fun(Y_prediction, Y[:,:self.n_out])

            #print('Accuracy of the network on test data: %f' % cost.item())
            #print(" [*] Testing finished!")
            

    def predict(self, X):
        self.model.eval()
        X = torch.from_numpy(X).float()   
        with torch.no_grad():             
            Y_prediction = self.model(X)         
            return Y_prediction
        
class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

use_cols = ['Год', 'Субъект', 'Численность населения']

args = Dict(lr = 0.01, epoch = 1000, n_in = n_steps_in, n_out = n_steps_out,
                num_centers = 100, testset = False,
                model_name='RBFN', cuda=False)

if Path('./data/demographics_forecast_py.csv').is_file():
    demographics_forecast = pd.read_csv('./data/demographics_forecast_py.csv', sep=";", decimal=',', encoding='cp1251')
    working_df = demographics_forecast[demographics_forecast['Базовый год'] == base_year][use_cols+['Базовый год']].reset_index(drop=True)
    if working_df.shape[0] == 0:
        working_df = demographics[demographics['Год'] <= base_year][use_cols].reset_index(drop=True)
        working_df['Базовый год'] = base_year
else:
    demographics_forecast = pd.DataFrame(columns=use_cols+['Базовый год'])
    working_df = demographics[demographics['Год'] <= base_year][use_cols].reset_index(drop=True)
    working_df['Базовый год'] = base_year



scalers = {}
n_features = len(use_cols)-2
n_steps_in, n_steps_out = 5, 1

model_path = Path(f'./pytorch_models/RBFN_model_{base_year}.txt')

if model_path.is_file():
  rbfn = RBFN_TS(args)
  rbfn.model.load_state_dict(torch.load(model_path))
  for reg in working_df['Субъект'].unique():
    scalers[reg] = MinMaxScaler()
    scalers[reg].fit_transform(working_df[(working_df['Субъект'] == reg)&(working_df['Год'] <= base_year)].sort_values(by='Год').drop(columns=['Год', 'Субъект', 'Базовый год']))
    for t_period in range(t):
        if (t_period +1) not in (working_df[working_df['Субъект'] == reg]['Год'] - base_year-1).values:
            scaled_x = scalers[reg].transform(working_df[working_df['Субъект'] == reg][use_cols[2:]])[-n_steps_in:]
            if scaled_x.flatten().shape[0]%(n_steps_in* n_features) == 0:
                x_pred_line = scaled_x.reshape((1, n_steps_in, n_features))
                y_pred_line = rbfn.predict(x_pred_line)
                y_pred = scalers[reg].inverse_transform(y_pred_line)
                y_pred = np.clip(y_pred, 0, None).round()
                y_pred = np.append([base_year+t_period+1, reg], y_pred)
                working_df = pd.concat([working_df, pd.DataFrame([y_pred], columns=use_cols)], axis=0, ignore_index=True)

working_df['Базовый год'] = base_year
demographics_forecast = demographics_forecast.drop(index=demographics_forecast[demographics_forecast['Базовый год'] == base_year].index)
demographics_forecast = pd.concat([demographics_forecast, working_df], ignore_index=True)
demographics_forecast.to_csv('./data/demographics_forecast_py.csv', sep=';', encoding='cp1251', index=False, decimal=',')

forecast_df = forecast_df.merge(working_df[['Субъект', 'Год', 'Численность населения']], on=['Субъект', 'Год'], how='left').rename(columns={'Численность населения': 'RBFN'})

forecast_df.to_csv("./data/forecast_df.csv", sep=';', mode='a', encoding='cp1251', index=False, decimal=',', header=not os.path.exists("./data/forecast_df.csv"))
forecast = forecast_df.drop(columns=['Базовая численность'])
forecast['Субъект'] = region_encoder.inverse_transform(forecast['Субъект'])
forecast.to_csv('./data/forecast.csv', sep=';', mode='a', encoding='cp1251', index=False, decimal=',', header=not os.path.exists('./data/forecast.csv'))


