# -*- coding: utf-8 -*-
import os
from flask import Flask, request
import sys
sys.path.append(os.path.dirname(__file__) + '/..')
import config
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
import logging
from logging.handlers import SMTPHandler, RotatingFileHandler
from config import basedir 
from flask_mail import Mail
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from flask_babel import Babel, lazy_gettext as _l
import pandas as pd
import subprocess
from pathlib import Path

from sqlalchemy import MetaData
from flask_ngrok import run_with_ngrok

convention = {
    "ix": 'ix_%(column_0_label)s',
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=convention)



app = Flask(__name__)
run_with_ngrok(app)
bootstrap = Bootstrap(app)
app.config.from_object(Config)
db = SQLAlchemy(app, metadata=metadata)
mail = Mail(app)
migrate = Migrate(app, db)
moment = Moment(app)
babel = Babel(app)
login = LoginManager(app)
login.login_view = 'login'
login.login_message = _l('Please log in to access this page.')
from app.models import Region, RegionStructure



if not app.debug:
    if app.config['MAIL_SERVER']:
        auth = None
        if app.config['MAIL_USERNAME'] or app.config['MAIL_PASSWORD']:
            auth = (app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        secure = None
        if app.config['MAIL_USE_TLS']:
            secure = ()
        mail_handler = SMTPHandler(
            mailhost=(app.config['MAIL_SERVER'], app.config['MAIL_PORT']),
            fromaddr='no-reply@' + app.config['MAIL_SERVER'],
            toaddrs=app.config['ADMINS'], subject='SADI Failure',
            credentials=auth, secure=secure)
        mail_handler.setLevel(logging.ERROR)
        app.logger.addHandler(mail_handler)
    if app.config['LOG_TO_STDOUT']:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        app.logger.addHandler(stream_handler)
    else:    
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = RotatingFileHandler('logs/SADI.log', maxBytes=10240,
                                           backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
    app.logger.setLevel(logging.INFO)
    app.logger.info('SADI startup')

def get_locale():
    return request.accept_languages.best_match(app.config['LANGUAGES'])

babel.init_app(app, locale_selector=get_locale)


def add_children(row):
    region = Region.query.filter_by(okato_name = row.okato_name).first()
    if region:
        for col_structure in row.index[2:]:
            year = int(col_structure.split('_')[-1])
            if isinstance(row[col_structure], str):
                for child_region_name in row[col_structure].split(','):
                    child_region = Region.query.filter_by(okato_name = child_region_name).first()
                    if child_region:
                        exists = RegionStructure.query.filter_by(region_id=region.id, child_reg_id=child_region.id, year=year).first()
                        if not exists:
                            db.session.add(RegionStructure(region_id=region.id, child_reg_id=child_region.id, year=year))   
                            db.session.commit()

def upload_data():
    if not (db.session.execute(db.text("SELECT * FROM region")).first() or db.session.execute(db.text("SELECT * FROM region_structure")).first()):
        df = pd.read_csv("./analysis/data/Территории.csv", sep=";").reset_index()
        df.columns= ['id', 'okato_name', 'structure_1990', 'structure_2010', 'structure_2014','structure_2018']
        df[['id', 'okato_name']].to_sql('region', con=db.engine, if_exists = "replace", index=False)
        df.apply(lambda row: add_children(row), axis=1)
        app.logger.info('Region uploaded')
    
    if not db.session.execute(db.text("SELECT * FROM forecast")).first():
        df = pd.read_csv('./analysis/data/forecast.csv', sep=';', encoding='cp1251', decimal=',').reset_index()
        df.columns = ['id', 'year','region_id', 'period','ext_inc','ext_gr','ext_exp','movements','straight','gomp_curve', 'lstm', 'rbfn']
        df['region_id'] = df['region_id'].apply(lambda reg: Region.query.filter_by(okato_name=reg).first().id)
        df['base_year'] = df['year'] - df['period']
        df.to_sql('forecast', con=db.engine, if_exists = "replace", index=False)
        app.logger.info('Forecast uploaded')

    if not Path('./analysis/data/new_demo_df.csv').is_file():
        subprocess.call(["python", "parse_demo.py", basedir], cwd=os.path.join(basedir,'./analysis/'))

    if not db.session.execute(db.text("SELECT * FROM demo_info")).first():
        new_demo_df = pd.read_csv('./analysis/data/new_demo_df.csv', sep=';', encoding='cp1251', decimal=',').reset_index()
        new_demo_df.columns = ['id', 'year','region_id', 'sex', 'type','population', 'immigrant', 'emigrant', 'deaths', 'births', 'abortion', 
                            'deceased_infants', 'adults_ablebodied', 'young_disabled', 'elderly_disabled', 'marriages', 'divorces', 
                            'lifespan', 'area', 'real_income', 'unemployment', 'cpi', 'pensions', 'poor', 'av_population', 'correction', 
                            'population_growth_rate', 'aging_rate', 'potential_load_factor', 'pension_burden_rate', 'total_load_factor', 
                            'ryabtsev_rate', 'fertility_rate', 'special_fertility_rate', 'tfr', 'abortion_rate', 'mortality_rate', 'infant_mortality_rate', 
                            'vitality_index', 'chil_rate', 'natural_growth', 'natural_growth_rate', 'mig_turnover', 'mig_balance', 'immigrantion_rate', 
                            'emigrantion_rate', 'mig_turnover_rate', 'mig_balance_rate', 'eff_coeff_mig_turnover', 'relative_balance_of_mig', 'mig_load_factor', 
                            'total_increase', 'marriage_rate', 'special_marriage_coeff', 'divorce_rate', 'divorce_index', 'gross_reproduction_coeff', 
                            'net_reproduction_coeff', 'generaten_length', 'true_natural_growth_rate', 'age_compensation', 'density']
        new_demo_df['region_id'] = new_demo_df['region_id'].apply(lambda reg: Region.query.filter_by(okato_name=reg).first().id)
        new_demo_df = new_demo_df.drop(columns=['correction'])
        new_demo_df['chil_rate'] = new_demo_df['chil_rate']*100
        new_demo_df.to_sql('demo_info', con=db.engine, if_exists = "replace", index=False)
        app.logger.info('DemoInfo uploaded')

        
    if not db.session.execute(db.text("SELECT * FROM age_structure")).first():
        df = pd.read_csv('./analysis/data/age_structure_names.csv', sep=';', encoding='cp1251', decimal=',').reset_index()
        df.columns = ['id', 'age', 'region_id', 'type','year', 'population_f','population_m','population']
        df['region_id'] = df['region_id'].apply(lambda reg: Region.query.filter_by(okato_name=reg).first().id)
        df['type'] = df['type'].replace('все население', 'все').replace('городское население', 'город').replace('сельское население', 'село')
        df.to_sql('age_structure', con=db.engine, if_exists = "replace", index=False)
        app.logger.info('Age structure uploaded')

    if not db.session.execute(db.text("SELECT * FROM death_rate")).first():
        m_df = pd.read_csv('./analysis/data/m_df.csv', sep=';', encoding='cp1251', decimal=',').reset_index()
        m_df.columns = ['id', 'year','region_id', 'sex','type','age','value']
        m_df['region_id'] = m_df['region_id'].apply(lambda reg: Region.query.filter_by(okato_name=reg).first().id)
        m_df.to_sql('death_rate', con=db.engine, if_exists = "replace", index=False)
        app.logger.info('DeathRate uploaded')

    if not db.session.execute(db.text("SELECT * FROM birth_rate")).first():
        f_df = pd.read_csv('./analysis/data/f_df.csv', sep=';', encoding='cp1251', decimal=',').reset_index()
        f_df.columns = ['id', 'year','region_id', 'type','age','value']
        f_df['region_id'] = f_df['region_id'].apply(lambda reg: Region.query.filter_by(okato_name=reg).first().id)
        f_df.to_sql('birth_rate', con=db.engine, if_exists = "replace", index=False)
        app.logger.info('BirthRate uploaded')

    app.logger.info('SADI data uploaded')

with app.app_context():
    db.create_all()
    upload_data()


  
from app import routes, models, errors

