from app import app
from app import db
from app.models import User, Post
import os, click

@app.cli.group()
def translate():
    """Translation and localization commands."""
    pass

@translate.command()
@click.argument('lang')
def init(lang):
    """Initialize a new language."""
    if os.system('pybabel extract -F babel.cfg -k _l -o messages.pot .'):
        raise RuntimeError('extract command failed')
    if os.system(f'pybabel init -i messages.pot -d app/translations -l {lang}'):
        raise RuntimeError('init command failed')
    os.remove('messages.pot')

@translate.command()
def update():
    """Update all languages."""
    if os.system('pybabel extract -F babel.cfg -k _l -o messages.pot .'):
        raise RuntimeError('extract command failed')
    if os.system('pybabel update -i messages.pot -d app/translations'):
        raise RuntimeError('update command failed')
    os.remove('messages.pot')

@translate.command()
def compile():
    """Compile all languages."""
    if os.system('pybabel compile -d app/translations'):
        raise RuntimeError('compile command failed')

@app.cli.group()
def user():
    """User commands."""
    pass

@user.command()
@click.argument('name')
def del_user(name):
    """Delete user."""
    u = User.query.filter_by(username = name).first()
    if u is not None:
        db.session.delete(u)
        db.session.commit()


@app.cli.group()
def post():
    """Post commands."""
    pass

@post.command()
@click.argument('id')
def del_post(id):
    """Delete post."""
    p = Post.query.filter_by(id = id).first()
    if p is not None:
        db.session.delete(p)
        db.session.commit()