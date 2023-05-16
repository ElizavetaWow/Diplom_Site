from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField, IntegerField, SelectField, SelectMultipleField
from wtforms.validators import ValidationError, DataRequired, Email, EqualTo, Length, NumberRange
from app.models import User
from flask_babel import _, lazy_gettext as _l
from app.models import Region, DemoInfo, Forecast

class RegistrationForm(FlaskForm):
    username = StringField(_l('Username'), validators=[DataRequired()])
    email = StringField(_l('Email'), validators=[DataRequired(), Email()])
    password = PasswordField(_l('Password'), validators=[DataRequired()])
    password2 = PasswordField(_l('Repeat Password'), validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField(_l('Register'))

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError(_('Please use a different username.'))

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError(_('Please use a different email address.'))

 
class LoginForm(FlaskForm):
    username = StringField(_l('Username'), validators=[DataRequired()])
    password = PasswordField(_l('Password'), validators=[DataRequired()])
    remember_me = BooleanField(_l('Remember Me'))
    submit = SubmitField(_l('Sign In'))
 

class EditProfileForm(FlaskForm):
    username = StringField(_l('Username'), validators=[DataRequired()])
    about_me = TextAreaField(_l('About me'), validators=[Length(min=0, max=40)])
    submit = SubmitField(_l('Submit'))
    cancel = SubmitField(_l('Cancel'))

    def __init__(self, original_username, *args, **kwargs):
        super(EditProfileForm, self).__init__(*args, **kwargs)
        self.original_username = original_username

    def validate_username(self, username):
        if username.data != self.original_username:
            user = User.query.filter_by(username=self.username.data).first()
            if user is not None:
                raise ValidationError(_('Please use a different username.'))

    
class ResetPasswordRequestForm(FlaskForm):
    email = StringField(_l('Email'), validators=[DataRequired(), Email()])
    submit = SubmitField(_l('Request Password Reset'))


class ResetPasswordForm(FlaskForm):
    password = PasswordField(_l('Password'), validators=[DataRequired()])
    password2 = PasswordField(
        _l('Repeat Password'), validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField(_l('Request Password Reset'))


class PostForm(FlaskForm):
    post = TextAreaField(_l('Say something'), validators=[DataRequired()])
    submit = SubmitField(_l('Submit'))


class ForecastForm(FlaskForm):
    base_year = IntegerField(_l('Base year'))
    period = IntegerField(_l('Forecast period'))
    region = SelectMultipleField(_l('Regions'), coerce=int)
    submit = SubmitField(_l('Forecast')) 


def edit_region(max_base_year, max_period, min_base_year):
    regions = Region.query.order_by(Region.okato_name.asc()).all()
    form = ForecastForm()
    form.region.choices = [(reg.id, reg.okato_name) for reg in regions]
    form.region.choices.insert(0, (-1, 'All'))
    form.region.default = [Region.query.filter(Region.okato_name == 'Российская Федерация').first().id]
    form.base_year.validators = [NumberRange(min=int(min_base_year), max=int(max_base_year)), DataRequired()]
    form.period.validators = [NumberRange(min=1, max=int(max_period)), DataRequired()]
    
    return form

class DemoInfoForm(FlaskForm):
    sex = SelectField(_l('Sex'), choices=[('ж', _l('Female')),('м', _l('Мale')),('все', _l('Both'))], validators=[DataRequired()], default='все')
    type = SelectField(_l('Types'), choices=[ ('город', _l('Urban population')),('село', _l('Rural population')),('все', _l('All population'))], validators=[DataRequired()], default='все')
    region = SelectMultipleField(_l('Regions'), coerce=int, validators=[DataRequired()])
    submit = SubmitField(_l('Find')) 

def edit_demoinfo():
    regions = Region.query.order_by(Region.okato_name.asc()).all()
    form = DemoInfoForm()
    form.region.choices = [(reg.id, reg.okato_name) for reg in regions]
    form.region.choices.insert(0, (-1, 'All'))
    form.region.default = [Region.query.filter(Region.okato_name == 'Российская Федерация').first().id]
    return form

class ChartForm(FlaskForm):
    category =  SelectField(_l('Category'), choices=[('year', _l('Year')),('sex', _l('Sex')),('type', _l('Type')),('region_id', _l('Region'))], validators=[DataRequired()], default='year')
    category2 =  SelectField(_l('Category2'), choices=[('year', _l('Year')),('sex', _l('Sex')),('type', _l('Type')),('region_id', _l('Region'))], validators=[DataRequired()], default='region_id')
    
    params = SelectMultipleField(_l('Parameters'), validators=[DataRequired()])
    year = SelectMultipleField(_l('Year'), coerce=int, validators=[DataRequired()])
    sex = SelectMultipleField(_l('Sex'), choices=[('ж', 'Женский'),('м', 'Мужской'),('все', 'Оба пола')], validators=[DataRequired()])
    type = SelectMultipleField(_l('Type'), choices=[ ('город', 'Городское население'),('село', 'Сельское население'),('все', 'Всё население')], validators=[DataRequired()])
    region_id = SelectMultipleField(_l('Region'), coerce=int, validators=[DataRequired()])

    chart_title = StringField(_l('Title'), validators=[DataRequired()], default=_l('Title'))
    chart_subtitle = StringField(_l('Subtitle'), validators=[DataRequired()], default=_l('Subtitle'))
    chart_type = SelectField(_l('Chart type'), choices=[('line', _l('Line graph')),('area', _l('Area chart')),('bar', _l('Bar chart')),
                                                        ('stacked_bar', _l('Stacked bar chart')),('scatter', _l('Scatter plot')), 
                                                        ('column', _l('Column chart')),('stacked_column', _l('Stacked column chart'))], validators=[DataRequired()], default='line')
    
    forecast_check = BooleanField(_l('Add forecast data'))
    base_year = IntegerField(_l('Base year'))
    period = IntegerField(_l('Forecast period'))
    forecast_type = SelectMultipleField(_l('Forecast type'))
    
    submit = SubmitField(_l('Draw')) 


def edit_chart(max_base_year, max_period, min_base_year):
    regions = Region.query.order_by(Region.okato_name.asc()).all()
    form = ChartForm()
    form.region_id.choices = [(reg.id, reg.okato_name) for reg in regions]
    years = DemoInfo.query.with_entities(DemoInfo.year).distinct()
    form.year.choices = [(int(y[0]), y[0]) for y in years]

    params = DemoInfo.query.first().names_dict
    form.category.choices.extend(list(params.items()))

    form.params.choices = list(params.items())
    form.params.choices.append(('age_structure', 'Половозрастная структура'))

    form.base_year.validators = [NumberRange(min=int(min_base_year), max=int(max_base_year))]
    form.period.validators = [NumberRange(min=1, max=int(max_period))]

    type_names = Forecast.query.first().type_names
    form.forecast_type.choices = list(type_names.items())
    return form
