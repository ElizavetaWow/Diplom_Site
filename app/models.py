from app import db
from app import login
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from hashlib import md5
from datetime import datetime
from time import time
import jwt
from app import app
from sqlalchemy import MetaData


metadata_obj = MetaData()


class Forecast(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer)
    base_year = db.Column(db.Integer)
    region_id = db.Column(db.Integer, db.ForeignKey('region.id'))
    period = db.Column(db.Integer)
    ext_inc = db.Column(db.Integer)
    ext_gr = db.Column(db.Integer)
    ext_exp = db.Column(db.Integer)
    movements = db.Column(db.Integer)
    straight = db.Column(db.Integer)
    gomp_curve = db.Column(db.Integer)
    lstm = db.Column(db.Integer)
    rbfn = db.Column(db.Integer)
    type_names = {'ext_inc':'Экстраполяция по приросту',
                  'ext_gr':'Экстраполяция по темпу роста',
                'ext_exp':'Экстраполяция по экспоненте',
                'movements':'Передвижки',
                'straight':'Выравнивание по прямой', 
                'gomp_curve':'Кривая роста Гомперца',
                'lstm':'LSTM', 'rbfn':'RBFN'}


@login.user_loader
def load_user(id):
    return User.query.get(int(id))


followers = db.Table('followers',
   db.Column('follower_id', db.Integer, db.ForeignKey('user.id')),
     db.Column('followed_id', db.Integer, db.ForeignKey('user.id'))
)

class RegionStructure(db.Model):
    __tablename__ = "region_structure"
    region_id = db.Column(db.Integer, db.ForeignKey("region.id"), primary_key=True)
    child_reg_id = db.Column(db.Integer, db.ForeignKey("region.id"), primary_key=True)
    year = db.Column(db.Integer, primary_key=True) 


class Region(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    okato_name =  db.Column(db.String(64))
    children = db.relationship('Region', secondary='region_structure',
                                primaryjoin=(RegionStructure.region_id == id),
                                secondaryjoin=(RegionStructure.child_reg_id == id),
                                backref=db.backref('parent', lazy='dynamic'), lazy='dynamic')
    
    death_rates = db.relationship('DeathRate', backref='region', lazy='dynamic') 
    birth_rates = db.relationship('BirthRate', backref='region', lazy='dynamic') 
    demo_infos = db.relationship('DemoInfo', backref='region', lazy='dynamic') 
    forecasts = db.relationship('Forecast', backref='region', lazy='dynamic') 
    
    def add_child(self, child, year):
        self.children.append(child, year)

class AgeStructure(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer)
    region_id = db.Column(db.Integer, db.ForeignKey('region.id'))
    type = db.Column(db.String(64))
    year = db.Column(db.Integer)
    population_f = db.Column(db.Integer)
    population_m = db.Column(db.Integer)
    population = db.Column(db.Integer)

class DeathRate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer)
    region_id = db.Column(db.Integer, db.ForeignKey('region.id'))
    sex = db.Column(db.String(64))
    type = db.Column(db.String(64))
    age = db.Column(db.String(64))
    value = db.Column(db.Float)

class BirthRate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer)
    region_id = db.Column(db.Integer, db.ForeignKey('region.id'))
    type = db.Column(db.String(64))
    age = db.Column(db.String(64))
    value = db.Column(db.Float)


class DemoInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer)
    region_id = db.Column(db.Integer, db.ForeignKey('region.id'))
    sex = db.Column(db.String(64))
    type = db.Column(db.String(64))
    
    population = db.Column(db.Integer)
    immigrant = db.Column(db.Integer)
    emigrant = db.Column(db.Integer)
    deaths = db.Column(db.Integer)
    births = db.Column(db.Integer)
    abortion = db.Column(db.Integer)
    deceased_infants = db.Column(db.Integer)
    adults_ablebodied = db.Column(db.Integer)
    young_disabled = db.Column(db.Integer)
    elderly_disabled = db.Column(db.Integer)
    marriages = db.Column(db.Integer)
    divorces = db.Column(db.Integer)
    lifespan = db.Column(db.Float)
    area = db.Column(db.Float)
    real_income = db.Column(db.Float)
    unemployment = db.Column(db.Float)
    cpi = db.Column(db.Float)
    pensions = db.Column(db.Float)
    poor = db.Column(db.Integer)
    av_population = db.Column(db.Integer)
    population_growth_rate = db.Column(db.Float)
    aging_rate = db.Column(db.Float)
    potential_load_factor = db.Column(db.Float)
    pension_burden_rate = db.Column(db.Float)
    total_load_factor = db.Column(db.Float)
    ryabtsev_rate = db.Column(db.Float)
    fertility_rate = db.Column(db.Float)
    special_fertility_rate = db.Column(db.Float)
    tfr = db.Column(db.Float)
    abortion_rate = db.Column(db.Float)
    mortality_rate = db.Column(db.Float)
    infant_mortality_rate = db.Column(db.Float)
    vitality_index = db.Column(db.Float)
    chil_rate = db.Column(db.Float)
    natural_growth = db.Column(db.Integer)
    natural_growth_rate = db.Column(db.Float)
    mig_turnover = db.Column(db.Integer)
    mig_balance = db.Column(db.Integer)
    immigrantion_rate = db.Column(db.Float)
    emigrantion_rate = db.Column(db.Float)
    mig_turnover_rate = db.Column(db.Float)
    mig_balance_rate = db.Column(db.Float)
    eff_coeff_mig_turnover = db.Column(db.Float)
    relative_balance_of_mig = db.Column(db.Float)
    mig_load_factor = db.Column(db.Float)
    total_increase = db.Column(db.Integer)
    marriage_rate = db.Column(db.Float)
    special_marriage_coeff = db.Column(db.Float)
    divorce_rate = db.Column(db.Float)
    divorce_index = db.Column(db.Float)
    gross_reproduction_coeff = db.Column(db.Float)
    net_reproduction_coeff = db.Column(db.Float)
    generaten_length = db.Column(db.Float)
    true_natural_growth_rate = db.Column(db.Float)
    age_compensation = db.Column(db.Float)
    density = db.Column(db.Float)
    names_dict = {'real_income': 'Реальные денежные доходы', 'unemployment': 'Уровень безработицы', 'cpi': 'Индекс потребительских цен', 'pensions': 'Реальный размер назначенных пенсий', 'poor': 'Численность малоимуших', 'density': 'Плотность', 'marriages': 'Число браков', 'divorces': 'Число разводов', 'marriage_rate': 'Коэффициент брачности', 'special_marriage_coeff': 'Специальный коэффициент брачности', 'divorce_rate': 'Коэффициент разводимости', 'divorce_index': 'Индекс разводимости', 'adults_ablebodied': 'Численность взрослое трудоспособное', 'young_disabled': 'Численность молодое нетрудоспособное', 'elderly_disabled': 'Численность пожилое нетрудоспособное', 'aging_rate': 'Уровень демографической старости', 'potential_load_factor': 'Коэффициент потенциального нагрузки', 'pension_burden_rate': 'Коэффициент пенсионной нагрузки', 'total_load_factor': 'Коэффициент общей нагрузки', 'ryabtsev_rate': 'Коэффициент Рябцева', 'immigrant': 'Число прибывших', 'emigrant': 'Число выбывших', 'mig_turnover': 'Миграционный оборот', 'immigrantion_rate': 'Коэффициент прибытия', 'emigrantion_rate': 'Коэффициент выбытия', 'mig_turnover_rate': 'Коэффициент миграционного оборота', 'mig_balance_rate': 'Коэффициент миграционного сальдо', 'eff_coeff_mig_turnover': 'Коэффициент эффективности миграционного оборота', 'relative_balance_of_mig': 'Относительное сальдо миграции', 'mig_load_factor': 'Коэффициент миграционной нагрузки', 'deaths': 'Число умерших', 'deceased_infants': 'Число умерших младенцев', 'lifespan': 'Продолжительность жизни', 'mortality_rate': 'Коэффициент смертности', 'infant_mortality_rate': 'Коэффициент младенческой смертности', 'vitality_index': 'Индекс жизненности', 'births': 'Число родившихся', 'abortion': 'Число абортов', 'fertility_rate': 'Коэффициент рождаемости', 'special_fertility_rate': 'Специальный коэффициент рождаемости', 'tfr': 'TFR', 'abortion_rate': 'Коэффициент прерывания беременности', 'chil_rate': 'Коэффициент детности', 'population': 'Численность', 'area': 'Площадь', 'av_population': 'Средняя численность', 'population_growth_rate': 'Темп прироста численности', 'natural_growth': 'Естественный прирост', 'natural_growth_rate': 'Коэффициент естественного прироста', 'mig_balance': 'Миграционное сальдо', 'total_increase': 'Общий прирост', 'gross_reproduction_coeff': 'Брутто-коэффициент воспроизводства', 'net_reproduction_coeff': 'Нетто-коэффициент воспроизводства', 'generaten_length': 'Длина поколения', 'true_natural_growth_rate': 'Истинный коэффициент естественного прироста', 'age_compensation': 'Компенсация возрастной структурой'}
    measurements_dict = {'real_income': '%', 'unemployment': '%', 'cpi': '%', 'pensions': '%', 'poor': 'чел.', 'density': 'чел/км^2', 'marriages': 'шт', 'divorces': 'шт', 'marriage_rate': '‰', 'special_marriage_coeff': '‰', 'divorce_rate': '‰', 'divorce_index': '%', 'adults_ablebodied': 'чел.', 'young_disabled': 'чел.', 'elderly_disabled': 'чел.', 'aging_rate': '%', 'potential_load_factor': '%', 'pension_burden_rate': '%', 'total_load_factor': '%', 'ryabtsev_rate': 'доля', 'immigrant': 'чел.', 'emigrant': 'чел.', 'mig_turnover': 'чел.', 'immigrantion_rate': '‰', 'emigrantion_rate': '‰', 'mig_turnover_rate': '‰', 'mig_balance_rate': '‰', 'eff_coeff_mig_turnover': '%', 'relative_balance_of_mig': '%', 'mig_load_factor': '‰', 'deaths': 'чел.', 'deceased_infants': 'чел.', 'lifespan': 'лет', 'mortality_rate': '‰', 'infant_mortality_rate': '‰', 'vitality_index': '%', 'births': 'чел.', 'abortion': 'шт', 'fertility_rate': '‰', 'special_fertility_rate': '‰', 'tfr': '‰', 'abortion_rate': '‰', 'chil_rate': '%', 'population': 'чел.', 'area': 'км^2', 'av_population': 'чел.', 'population_growth_rate': '%', 'natural_growth': 'чел.', 'natural_growth_rate': '‰', 'mig_balance': 'чел.', 'total_increase': 'чел.', 'gross_reproduction_coeff': '‰', 'net_reproduction_coeff': '‰', 'generaten_length': 'лет', 'true_natural_growth_rate': '‰', 'age_compensation': '‰'}


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    posts = db.relationship('Post', backref='author', lazy='dynamic')
    about_me = db.Column(db.String(140))
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    followed = db.relationship(
        'User', secondary=followers,
        primaryjoin=(followers.c.follower_id == id),
        secondaryjoin=(followers.c.followed_id == id),
        backref=db.backref('followers', lazy='dynamic'), lazy='dynamic')

    def follow(self, user):
        if not self.is_following(user):
            self.followed.append(user)

    def unfollow(self, user):
        if self.is_following(user):
            self.followed.remove(user)

    def is_following(self, user):
        return self.followed.filter(
            followers.c.followed_id == user.id).count() > 0
  
    def followed_posts(self):
        followed = Post.query.join(
            followers, (followers.c.followed_id == Post.user_id)).filter(
                followers.c.follower_id == self.id)
        own = Post.query.filter_by(user_id=self.id)
        return followed.union(own).order_by(Post.timestamp.desc())       


    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
  
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
 
    def get_reset_password_token(self, expires_in=600):
        return jwt.encode(
            {'reset_password': self.id, 'exp': time() + expires_in},
            app.config['SECRET_KEY'], algorithm='HS256').decode('utf-8')

    @staticmethod
    def verify_reset_password_token(token):
        # sourcery skip: avoid-builtin-shadow
        try:
            id = jwt.decode(token, app.config['SECRET_KEY'],
                            algorithms=['HS256'])['reset_password']
        except Exception:
            return
        return User.query.get(id)


    def avatar(self, size):
        digest = md5(self.email.lower().encode('utf-8')).hexdigest()
        return f'https://www.gravatar.com/avatar/{digest}?d=identicon&s={size}'


    def __repr__(self):
        return f'<User {self.username}>'


class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    body = db.Column(db.String(140))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    language = db.Column(db.String(5))
    
    def __repr__(self):
        return f'<Post {self.body}>'