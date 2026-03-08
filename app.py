import os
from flask import Flask, render_template, request, url_for, redirect
from flask import Flask, request, jsonify, render_template, redirect, url_for, session

from werkzeug.utils import secure_filename
import librosa as lb
import librosa.display
import matplotlib
#  to avoid flask err of RuntimeError: main thread is not in main loop
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rdc_model



root_folder = os.path.abspath(os.path.dirname(__file__))
print(root_folder)
UPLOAD_FOLDER_temp = os.path.join(root_folder, "static")
UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER_temp,"uploads")
print(UPLOAD_FOLDER)
app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



users = {}

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['username'] = username  # Set the session variable
            return redirect(url_for('index'))
        else:
            return "Login failed. Please check your username and password."

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username not in users:
            users[username] = password
            return redirect(url_for('login'))
        else:
            return "Registration failed. User already exists."

    return render_template('register.html')


@app.route("/index")
def index():
    dir = UPLOAD_FOLDER
    # empty uploads folder as we do not save sound files of patients
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    return render_template("index.html",ospf = 1)

@app.route("/", methods = ['POST'])
def patient():
    if request.method == "POST":
        # imp to clear matplotlib cache else it will save the previous figure
        plt.figure().clear()
        
        print(request)
        name = request.form["name"] #taking data from dictionary
        lungSounds = request.files["lungSounds"]
        print("\n")
        filename = secure_filename(lungSounds.filename)
        # temporarily save sound file of patient in the Uploads folder
        lungSounds.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        url2 = os.path.join("static", "uploads")
        url = os.path.join(url2, filename)
        # url = os.path.abspath(url)
        print(url)
        absolute_url =  os.path.abspath(url)
        
        # pass url of sound file to the model
        res_list = rdc_model.classificationResults(absolute_url)

        # librosa can convert stereo to mono audio
        audio1,sample_rate1 = lb.load(url,  mono=True)

        soundWave = librosa.display.waveshow(audio1,sr=sample_rate1, max_points=50000, x_axis='time', offset=0)
        # save python plot img
        plt.savefig("./static/uploads/outSoundWave.png")
        
        mfccs = lb.feature.mfcc(y=audio1, sr=sample_rate1, n_mfcc=40)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax)
        plt.savefig("./static/uploads/outSoundMFCC.png")

        url3 = os.path.join(url2,"outSoundWave.png")
        print(url3)
        res_list.append(os.path.abspath(url3))

    return render_template("index.html",ospf = 0,n = name,  lungSounds = url, res = res_list)


@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/piechart')
def piechart():
    return render_template('piechart.html')



if __name__ == "__main__":
    app.run()