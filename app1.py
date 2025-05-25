from flask import Flask, render_template, flash, request, session, send_file
from flask import render_template, redirect, url_for, request

import sys

import pickle

import numpy as np

app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def homepage():
    return render_template('home.html')


@app.route("/result", methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        import tensorflow as tf
        import numpy as np

        t1 = request.form['t1']
        t2 = request.form['t2']
        t22 = request.form['t22']
        t23 = request.form['t23']
        t24 = request.form['t24']
        t25 = request.form['t25']
        t26 = request.form['t26']
        t27 = request.form['t27']
        t28 = request.form['t28']
        t29 = request.form['t29']
        t210 = request.form['t210']
        t211 = request.form['t211']
        t212 = request.form['t213']
        t213 = request.form['t213']
        t214 = request.form['t214']
        t215 = request.form['t215']
        t216 = request.form['t216']
        t217 = request.form['t217']
        t218 = request.form['t218']
        t219 = request.form['t219']
        t220 = request.form['t220']
        t221 = request.form['t221']
        t222 = request.form['t222']
        t223 = request.form['t223']
        t224 = request.form['t224']
        t225 = request.form['t225']
        t226 = request.form['t226']
        t227 = request.form['t227']
        t228 = request.form['t228']
        t229 = request.form['t229']
        t230 = request.form['t230']
        t231 = request.form['t231']
        t232 = request.form['t232']
        t233 = request.form['t233']
        t234 = request.form['t234']
        t235 = request.form['t235']
        t236 = request.form['t236']
        t237 = request.form['t237']

        t1 = float(t1)
        t2 = float(t2)
        t22 = float(t22)
        t23 = float(t23)
        t24 = float(t24)
        t25 = float(t25)
        t26 = float(t26)
        t27 = float(t27)
        t28 = float(t28)
        t29 = float(t29)
        t210 = float(t210)
        t211 = float(t211)
        t212 = float(t213)
        t213 = float(t213)
        t214 = float(t214)
        t215 = float(t215)
        t216 = float(t216)
        t217 = float(t217)
        t218 = float(t218)
        t219 = float(t219)
        t220 = float(t220)
        t221 = float(t221)
        t222 = float(t222)
        t223 = float(t223)
        t224 = float(t224)
        t225 = float(t225)
        t226 = float(t226)
        t227 = float(t227)
        t228 = float(t228)
        t229 = float(t229)
        t230 = float(t230)
        t231 = float(t231)
        t232 = float(t232)
        t233 = float(t233)
        t234 = float(t234)
        t235 = float(t235)
        t236 = float(t236)
        t237 = float(t237)

        model = tf.keras.models.load_model('model.h5')
        data = np.array([[t1, t2, t22, t23, t24, t25, t26, t27, t28, t29, t210,
                          t211,
                          t212,
                          t213,
                          t214,
                          t215,
                          t216,
                          t217,
                          t218,
                          t219,
                          t220,
                          t221, t222, t223, t224, t225, t226, t227, t228, t229, t230,
                          t231, t232, t233, t234, t235, t236, t237
                          ]])

        my_prediction1 = model.predict(data, batch_size=64)

        my_prediction = np.argmax(my_prediction1)
        print(my_prediction)

        print(my_prediction)
        Answer = ''

        if my_prediction == 0:
            Answer = 'back'

        elif my_prediction == 1:
            Answer = 'ftp_write'
        elif my_prediction == 2:
            Answer = 'R2L'
        elif my_prediction == 3:
            Answer = 'smurf'
        elif my_prediction == 4:
            Answer = 'normal'
        elif my_prediction == 5:
            Answer = 'multihop'
        elif my_prediction == 6:
            Answer = 'probing'
        elif my_prediction == 7:
            Answer = 'Accident'

        return render_template('home.html', res=Answer)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
