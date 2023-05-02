from app import app, db, cli
from app.models import User, Post, Forecast, DemoInfo, BirthRate, DeathRate, AgeStructure, Region, RegionStructure


@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'Post': Post, 'Forecast':Forecast, 'DemoInfo':DemoInfo, 'BirthRate':BirthRate, 
            'DeathRate':DeathRate, 'AgeStructure':AgeStructure, 'Region':Region,'RegionStructure':RegionStructure}
