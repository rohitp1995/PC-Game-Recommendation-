import flask
from model import check_game, get_recommendation,all_games

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():

    no_of_recommendation = 5

    if flask.request.method == 'GET':
        return(flask.render_template('index.html',game_list=all_games))

    if flask.request.method == 'POST':
        g_name = flask.request.form['game_name']
        if not check_game(g_name):
            return(flask.render_template('not found.html',name=g_name))
        else:
            names = get_recommendation(g_name, no_of_recommendation+1)
            return flask.render_template('found.html',game_names=names)


if __name__ == '__main__':
    app.run()   
