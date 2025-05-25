from flask import Flask, render_template, flash, request, session, send_file
from flask import render_template, redirect, url_for, request
import os
import mysql.connector
app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/Prediction")
def Prediction():
    return render_template('Prediction.html')



@app.route('/UserLogin')
def UserLogin():
    return render_template('UserLogin.html')


@app.route("/NewUser")
def NewUser():
    return render_template('NewUser.html')



@app.route("/newuser", methods=['GET', 'POST'])
def newuser():
    if request.method == 'POST':
        name = request.form['name']

        age = request.form['age']
        mobile = request.form['mobile']
        email = request.form['email']
        address = request.form['address']
        username = request.form['username']
        Password = request.form['Password']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='5copymovedb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where username='" + username + "' ")
        data = cursor.fetchone()
        if data is None:

            conn = mysql.connector.connect(user='root', password='', host='localhost', database='5copymovedb')
            cursor = conn.cursor()
            cursor.execute(
                "insert into regtb values('','" + name + "','" + age + "','" + mobile + "','" + email + "','" + address + "','" + username + "','" + Password + "')")
            conn.commit()
            conn.close()
            return render_template('UserLogin.html')



        else:
            flash('Already Register Username')
            return render_template('NewUser.html')




@app.route("/userlogin", methods=['GET', 'POST'])
def userlogin():
    if request.method == 'POST':

        username = request.form['uname']
        password = request.form['Password']
        session['uname'] = request.form['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='5copymovedb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where username='" + username + "' and Password='" + password + "'")
        data = cursor.fetchone()
        if data is None:

            flash('Username or Password is wrong')
            return render_template('UserLogin.html')

        else:
            mobil = data[3]
            session['mobil'] = mobil

            return render_template('Prediction.html')

           
        





@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        import tensorflow as tf
        import numpy as np
        from tkinter import messagebox

        model = tf.keras.models.load_model('model.h5')

        float_array = np.fromstring(name, dtype=float, sep=',')

        print(float_array)

        reshaped_array = float_array.reshape(1, -1)

        pred = model.predict(reshaped_array, batch_size=64)
        print(pred)


        my_prediction = np.argmax(pred,axis=1)
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

        #sendmsg(session['mobil'],"Result : " +Answer)





        return render_template('Result.html', result=Answer)




def sendmsg(targetno,message):
    import requests
    requests.post(
        "http://sms.creativepoint.in/api/push.json?apikey=6555c521622c1&route=transsms&sender=FSSMSS&mobileno=" + targetno + "&text=Dear customer your msg is " + message + "  Sent By FSMSG FSSMSS")




if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
