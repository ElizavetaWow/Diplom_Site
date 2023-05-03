from flask import render_template, flash, redirect, url_for, request, g
from app import app, db
from app.forms import LoginForm, RegistrationForm, EditProfileForm, PostForm, edit_region, Region, edit_demoinfo, edit_chart
from app.forms import ResetPasswordRequestForm
from app.forms import ResetPasswordForm
from app.models import User, Post, Forecast, DemoInfo, DeathRate, BirthRate, AgeStructure
from app.emailf import send_password_reset_email
from flask_babel import _, get_locale
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.urls import url_parse
from datetime import datetime
from guess_language import guess_language
from sqlalchemy import func
from itertools import compress
import pandas as pd


@app.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()
    g.locale = str(get_locale()) 



def check_subtable(items, cols_list):
    mask = []
    for i in range(len(items)):
        k = sum(getattr(items[i], col) is None for col in cols_list)
        if k == len(cols_list):
            mask.append(False)
        else:
            mask.append(True)
    return list(compress(items, mask)) 


@app.route('/demoinfo', methods = ['GET', 'POST'])
def demoinfo():
    regions = Region.query.all()
    form = edit_demoinfo()
    sex = 'все'
    typ = 'все'
    regs = [Region.query.filter(Region.okato_name == 'Российская Федерация').first().id]

    if form.validate_on_submit():
        sex = form.sex.data
        typ = form.type.data
        regs = form.region.data
        if -1 in regs:
            items = DemoInfo.query.filter(DemoInfo.type == typ, DemoInfo.sex == sex).all()
            death_items = DeathRate.query.filter(DeathRate.type == typ, DeathRate.sex == sex).all()
            birth_items = BirthRate.query.filter(BirthRate.type == typ).all()
        else:
            items = DemoInfo.query.filter(DemoInfo.type == typ, DemoInfo.sex == sex, DemoInfo.region_id.in_(regs)).all()
            death_items = DeathRate.query.filter(DeathRate.type == typ, DeathRate.sex == sex, DeathRate.region_id.in_(regs)).all()
            birth_items = BirthRate.query.filter(BirthRate.type == typ, BirthRate.region_id.in_(regs)).all()
    else:
        items = DemoInfo.query.filter(DemoInfo.type == typ, DemoInfo.sex == sex, DemoInfo.region_id.in_(regs)).all()
        death_items = DeathRate.query.filter(DeathRate.type == typ, DeathRate.sex == sex, DeathRate.region_id.in_(regs)).all()
        birth_items = BirthRate.query.filter(BirthRate.type == typ, BirthRate.region_id.in_(regs)).all()
    cols_general = ['av_population', 'population_growth_rate', 'natural_growth', 'natural_growth_rate', 'mig_balance', 
                     'total_increase', 'gross_reproduction_coeff', 'net_reproduction_coeff', 'generaten_length', 
                     'true_natural_growth_rate', 'age_compensation']
    cols_births = ['births', 'abortion', 'fertility_rate', 'special_fertility_rate', 'tfr', 'abortion_rate', 'chil_rate']
    cols_deaths = ['deaths', 'deceased_infants', 'lifespan', 'mortality_rate', 'infant_mortality_rate', 'vitality_index']
    cols_migration = ['immigrant', 'emigrant', 'mig_turnover', 'mig_balance', 'immigrantion_rate', 'emigrantion_rate', 'mig_turnover_rate', 'mig_balance_rate', 'eff_coeff_mig_turnover', 'relative_balance_of_mig', 'mig_load_factor']
    cols_structure =['adults_ablebodied', 'young_disabled', 'elderly_disabled', 'aging_rate', 'potential_load_factor', 'pension_burden_rate', 'total_load_factor', 'ryabtsev_rate']
    cols_marriage =['marriages', 'divorces', 'marriage_rate', 'special_marriage_coeff', 'divorce_rate', 'divorce_index']
    cols_economics =['real_income', 'unemployment', 'cpi', 'pensions', 'poor', 'density']
    rate_cols = ['age', 'value']

    items_general = check_subtable(items, cols_general)
    items_births = check_subtable(items, cols_births)
    items_deaths = check_subtable(items, cols_deaths)
    items_migration = check_subtable(items, cols_migration)
    items_structure = check_subtable(items, cols_structure)
    items_marriage = check_subtable(items, cols_marriage)
    items_economics = check_subtable(items, cols_economics)
    death_items = check_subtable(death_items, rate_cols)
    birth_items = check_subtable(death_items, rate_cols)
    
    return render_template('data_info.html',title = _('Data'), regions=regions, form=form, 
                           items_general=items_general, items_births=items_births, items_deaths=items_deaths, 
                           items_migration=items_migration, items_structure=items_structure, items_marriage=items_marriage,
                           items_economics=items_economics,death_items=death_items, birth_items=birth_items)


@app.route('/charts', methods = ['GET', 'POST'])
def charts(chartID = 'chart_ID', chart_type = 'line'):
    max_base_year = int(db.session.query(func.max(Forecast.base_year)).scalar())
    min_base_year = int(db.session.query(func.min(Forecast.base_year)).scalar())
    max_period = int(db.session.query(func.max(Forecast.period)).scalar())
    form = edit_chart(max_base_year, max_period, min_base_year)
    base_year = max_base_year
    period = 10
    tooltip = {}
    series = []
    title = {"text": 'My Title'}
    subtitle =  {"text": 'My Subtitle'}
    xAxis = {"categories": []}
    yAxis = [{"title": {"text": 'yAxis Label'}}]
    plotOptions={}
    if form.is_submitted():
        measurements_dict = DemoInfo.query.first().measurements_dict
        category = form.category.data
        category2 = form.category2.data
        chart_type = form.chart_type.data
        filts = {'year':form.year.data or [2022], 'sex':form.sex.data or ['все'], 'type':form.type.data or ['все'], 'region_id':form.region_id.data or [0], 'params':form.params.data or ['year']}       
        series = []
        current_measurements = []
        if form.base_year._value():
            base_year = int(form.base_year._value())
        if form.period._value():
            period = form.period._value()

        if len(filts['params']) == 1 and filts['params'][0] == 'age_structure':
            chart_type = 'bar'
            q = AgeStructure.query.filter(AgeStructure.type == filts['type'][0], 
                                          AgeStructure.region_id == filts['region_id'][0], 
                                          AgeStructure.year == filts['year'][0]).order_by(AgeStructure.age.desc())
            
            male_items = list(-pd.Series(list(zip(*q.with_entities(AgeStructure.population_m).all()))[0]))
            female_items = list(list(zip(*q.with_entities(AgeStructure.population_f).all()))[0])
            series.append({"name": _('Male'), "data": male_items})
            series.append({"name": _('Female'), "data": female_items})
            ages = list(list(zip(*q.with_entities(AgeStructure.age).all()))[0])
            ages_str= []
            ages_str.append(str(ages[0])+'+')
            for i in range(0, len(ages)-1):
                if abs(ages[i]-ages[i+1]) == 1:
                    ages_str.append(str(ages[i+1]))
                else:
                    ages_str.append(str(ages[i+1])+'-'+str(ages[i]-1))
            
            xAxis= [{'categories': ages_str,'reversed': 'false',}, 
                     {'opposite': 'true','reversed': 'false','categories': ages_str, 'linkedTo': 0}]
            yAxis = {"title": {"text": ''}, "labels":{"formatter":''}}
            tooltip = {"formatter" : "age_structure"}
        else:        
            if len(filts['params']) > 1 and 'age_structure' in filts['params']:
                filts['params'].remove('age_structure')
            if form.forecast_check.data:
                if max(filts['year']) < base_year:
                    filts['year'].extend(list(range(max(filts['year'])+1, base_year+1)))
            for cat2 in filts[category2]:
                clear_filts = filts.copy()
                clear_filts.pop(category2)
                if category in clear_filts.keys():
                        xAxis = {"categories": filts[category]}
                        clear_filts.pop(category)
                        clear_filts_keys = list(clear_filts.keys())

                        q = DemoInfo.query.filter(getattr(DemoInfo, category2) == cat2,  getattr(DemoInfo, category).in_(filts[category]), 
                                                getattr(DemoInfo, clear_filts_keys[0]) == clear_filts[clear_filts_keys[0]][0], 
                                                getattr(DemoInfo, clear_filts_keys[1]) == clear_filts[clear_filts_keys[1]][0])
                else:             
                    clear_filts_keys = list(clear_filts.keys())
                    q = DemoInfo.query.filter(getattr(DemoInfo, category2) == cat2, 
                                            getattr(DemoInfo, clear_filts_keys[0]) == clear_filts[clear_filts_keys[0]][0], 
                                            getattr(DemoInfo, clear_filts_keys[1]) == clear_filts[clear_filts_keys[1]][0]).order_by(getattr(DemoInfo, category).asc())
                    xAxis = { "title": {"text": dict(form.params.choices).get(category)}, "categories":  list(list(zip(*q.with_entities(getattr(DemoInfo, category)).all()))[0])}

                for param in filts['params']:
                    if param != 'age_structure':
                        items = list(list(zip(*q.with_entities(getattr(DemoInfo, param)).all()))[0])
                        if items.count(None) != len(items):
                            if len(filts['params']) == 1:
                                name = dict(getattr(form, category2).choices).get(cat2)
                            else:
                                name = dict(form.params.choices).get(param)
                            current_measurements.append(measurements_dict[param])
                            items = [('Null' if x is None else x) for x in items]
                            series.append({"name": name, "data": items})


            if form.forecast_check.data:
                
                for reg_id in filts['region_id']:
                    q = Forecast.query.filter(Forecast.base_year == base_year, Forecast.period <= period, Forecast.region_id == reg_id)
                    q2 = DemoInfo.query.filter(DemoInfo.type == filts['type'][0], DemoInfo.sex == filts['sex'][0], DemoInfo.region_id == reg_id, DemoInfo.year < int(base_year))
                    xAxis["categories"].extend(list(list(zip(*q.with_entities(Forecast.year).all())))[0])
                    for f_t in form.forecast_type.data:
                        items = list(list(zip(*q2.with_entities(DemoInfo.population).all()))[0])+list(list(zip(*q.with_entities(getattr(Forecast, f_t)).all()))[0])
                       
                        name = Region.query.filter(Region.id == reg_id).first().okato_name+'_'+dict(form.forecast_type.choices).get(f_t)
                        items = [('Null' if x is None else x) for x in items]
                        series.append({"name": name, "data": items})
      
            current_measurements_set = list(set(current_measurements))  
            yAxis= [{'title': {'text': 'Value'}}] 
            if len(current_measurements_set) == 1:
                yAxis[0]['title']['text'] = yAxis[0]['title']['text']+'in '+current_measurements_set[0]
            if len(current_measurements_set) == 2:        
                yAxis.append({'opposite': 'true','title': {'text': 'Value in '+current_measurements_set[1]}})
                for i in range(len(series)):
                    if current_measurements[i] == current_measurements_set[0]:
                        series[i]["yAxis"] = 0
                    else:
                        series[i]["yAxis"] = 1
        title = {"text": form.chart_title.data}
        subtitle =  {"text": form.chart_subtitle.data} 
        if len(series) == 0:
            flash(_('No data was found. Try to change chart settings.'))

             

    if chart_type == 'line':
        tooltip = {"shared": "true", "crosshairs": "true"}     
    elif chart_type in ['stacked_bar', 'stacked_column']:
        chart_type = chart_type.replace('stacked_', '')
        plotOptions= {'series': {'stacking': 'normal','dataLabels': {'enabled': 'true'}}}

    chart = {"renderTo": chartID, "type": chart_type,   "zoomType": 'xy'}
    
    return render_template('charts.html',title = _('Charts'), form = form, chartID=chartID, chart=chart, series=series, 
                           ctitle=title, xAxis=xAxis, yAxis=yAxis, subtitle=subtitle, tooltip=tooltip, base_year= base_year, period=period, plotOptions=plotOptions)




@app.route('/', methods = ['GET', 'POST'])
@app.route('/index', methods = ['GET', 'POST'])

def index():
    dataset = {}
    regs = Region.query.filter(Region.children == None).all()
    regs = [Region.query.filter(Region.okato_name == 'Российская Федерация').first()]+regs
    for reg in regs:
        dataset[reg.okato_name] = {}
        q = DemoInfo.query.filter(DemoInfo.type == 'все', DemoInfo.sex == 'все', DemoInfo.region_id == reg.id).all()
        for di in q:
            dataset[reg.okato_name][str(di.year)] = str(di.population)
    return render_template('index.html', title = _('SADI'), dataset=dataset) 



@app.route('/forecast', methods = ['GET', 'POST'])
def forecast():
    max_base_year = int(db.session.query(func.max(Forecast.base_year)).scalar())
    min_base_year = int(db.session.query(func.min(Forecast.base_year)).scalar())
    max_period = int(db.session.query(func.max(Forecast.period)).scalar())
    form = edit_region(max_base_year, max_period, min_base_year)
    base_year = max_base_year
    period = 10
    regs = [-1]
    if form.validate_on_submit():
        base_year = form.base_year._value()
        period = form.period._value()
        regs = form.region.data
        if -1 in regs:
            items = Forecast.query.filter_by(base_year = base_year, period=period).all()
        else:
            items = Forecast.query.filter(Forecast.base_year == base_year, Forecast.period <= period, Forecast.region_id.in_(regs)).all()
    else:
        items = Forecast.query.filter(Forecast.base_year == base_year, Forecast.period <= period).all()
    regions = Region.query.all()
    return render_template('forecast.html', title = _('Forecast'), form = form, items=items, regions=regions, 
                           base_year= base_year, period=period, regs=regs) 

@app.route('/blog', methods = ['GET', 'POST'])
def blog():
    form = PostForm()
    if form.validate_on_submit():
        language = guess_language(form.post.data)
        if language == 'UNKNOWN' or len(language) > 5:
            language = ''
        post = Post(body = form.post.data, author = current_user,
                    language = language)
        db.session.add(post)
        db.session.commit()
        flash(_('Your post is now live!'))
        return redirect(url_for('blog'))
    page = request.args.get('page', 1, type = int)
    posts = Post.query.order_by(Post.timestamp.desc()).paginate(
        page=page, per_page=app.config['POSTS_PER_PAGE'], error_out=False)
    next_url = (url_for('blog', page = posts.next_num)
                if posts.has_next else None)
    prev_url = (url_for('blog', page = posts.prev_num)
                if posts.has_prev else None)
    return render_template('blog.html', title = _('Blog'), form = form,
                           posts = posts.items, next_url = next_url,
                           prev_url = prev_url)


@app.route('/user/<username>', methods = ['GET', 'POST'])
@login_required
def user(username):
    form = PostForm()
    if form.validate_on_submit():
        language = guess_language(form.post.data)
        if language == 'UNKNOWN' or len(language) > 5:
            language = ''
        post = Post(body = form.post.data, author = current_user,
                    language = language)
        db.session.add(post)
        db.session.commit()
        flash(_('Your post is now live!'))
        return redirect(url_for('user', username = current_user.username))
    user = User.query.filter_by(username = username).first_or_404()
    page = request.args.get('page', 1, type = int)
    posts = user.posts.order_by(Post.timestamp.desc()).paginate(
       page=page, per_page=app.config['POSTS_PER_PAGE'], error_out=False)
    next_url = (url_for('user', username = user.username, page = posts.next_num)
                if posts.has_next else None)
    prev_url = (url_for('user', username = user.username, page = posts.prev_num)
                if posts.has_prev else None)
    return render_template('user.html', form = form, user = user,
                           posts = posts.items, next_url = next_url,
                           prev_url = prev_url, title = user.username)


@app.route('/edit_profile', methods = ['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm(current_user.username)
    if request.method == 'POST':
        if form.cancel.data:  
            return redirect(url_for('user', username = current_user.username))
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash(_('Your changes have been saved.'))
        return redirect(url_for('user', username = current_user.username))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', title = _('Edit Profile'),
                           form = form)


@app.route('/followers/<username>', methods = ['GET', 'POST'])
@login_required
def followers(username):
    user = User.query.filter_by(username = username).first()
    page = request.args.get('page', 1, type = int)
    followers = user.followers.order_by(User.username.asc()).paginate(
        page=page, per_page=app.config['POSTS_PER_PAGE'], error_out=False)
    next_url = (url_for('follow_ers_ings', username = user.username,
                        page = followers.next_num)
                if followers.has_next else None)
    prev_url = (url_for('follow_ers_ings', username = user.username,
                        page = followers.prev_num)
                if followers.has_prev else None)
    return render_template('follow_ers_ings.html', title = _('Followers'), 
                           user = user, folls = followers.items,
                           next_url = next_url, prev_url = prev_url)


@app.route('/followings/<username>', methods = ['GET', 'POST'])
@login_required
def followings(username):
    user = User.query.filter_by(username = username).first()
    page = request.args.get('page', 1, type = int)
    followings = user.followed.order_by(User.username.asc()).paginate(
       page=page, per_page=app.config['POSTS_PER_PAGE'], error_out=False)
    next_url = (url_for('follow_ers_ings', username = user.username,
                        page = followings.next_num)
                if followings.has_next else None)
    prev_url = (url_for('follow_ers_ings', username = user.username,
                        page = followings.prev_num)
                if followings.has_prev else None)
    return render_template('follow_ers_ings.html', title = _('Following'), 
                           folls = followings.items, next_url = next_url,
                           prev_url = prev_url, user = user)




@app.route('/follow/<username>')
@login_required
def follow(username):
    user = User.query.filter_by(username = username).first()
    if user is None:
        flash(_('User %(username)s not found.', username = username))
        return redirect(url_for('index'))
    if user == current_user:
        flash(_('You cannot follow yourself!'))
        return redirect(url_for('user', username = username))
    current_user.follow(user)
    db.session.commit()
    flash(_('You are following %(username)s!', username = username))
    return redirect(url_for('user', username = username))


@app.route('/unfollow/<username>')
@login_required
def unfollow(username):
    user = User.query.filter_by(username = username).first()
    if user is None:
        flash(_('User %(username)s not found.', username = username))
        return redirect(url_for('index'))
    if user == current_user:
        flash(_('You cannot unfollow yourself!'))
        return redirect(url_for('user', username = username))
    current_user.unfollow(user)
    db.session.commit()
    flash(_('You are not following %(username)s.', username = username))
    return redirect(url_for('user', username = username))


@app.route('/register', methods = ['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username = form.username.data, email = form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash(_('Congratulations, you are now a registered user!'))
        return redirect(url_for('login'))
    return render_template('register.html', title = _('Register'), form = form)


@app.route('/reset_password_request', methods = ['GET', 'POST'])
def reset_password_request():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = ResetPasswordRequestForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email = form.email.data).first()
        if user:
            send_password_reset_email(user)
        flash(
            _('Check your email for the instructions to reset your password'))
        return redirect(url_for('login'))
    return render_template('reset_password_request.html', form = form,
                           title = _('Reset Password'))


@app.route('/reset_password/<token>', methods = ['GET', 'POST'])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    user = User.verify_reset_password_token(token)
    if not user:
        return redirect(url_for('index'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        db.session.commit()
        flash(_('Your password has been reset.'))
        return redirect(url_for('login'))
    return render_template('reset_password.html', title = _('Reset password'),
                           form = form)


@app.route('/login', methods = ['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username = form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash(_('Invalid username or password'))
            return redirect(url_for('login'))
        login_user(user, remember = form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title = _('Sign In'), form = form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))