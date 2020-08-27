from flask import Flask, jsonify, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
from flask_restplus import Resource, Api
from multiprocessing import Value
import requests
import pandas as pd
import os
from src.src import NeuralNetwork

app = Flask(__name__, template_folder='templates')
api = Api(app)

SECRET_KEY = '12345'
app.secret_key = SECRET_KEY

class Status(Resource):
    def get(self):
        status = 'training...'
        return status
api.add_resource(Status, '/status')

UPLOAD_FOLDER = os.getcwd() + '/static/uploads/'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        f = request.files['inputs']

        # check if the post request has the file part
        if 'inputs' not in request.files:
            flash('No file in part')
        elif f.filename == '':
            flash('No selected file')
        elif not allowed_file(f.filename):
            flash(f'File not uploaded. File must be a csv')
            
        else:
            secure = secure_filename(f.filename)
            savename = os.path.join(app.config['UPLOAD_FOLDER'], secure)
            f.save(savename)
            flash(f'Successfully uploaded {f.filename}')
    
    return render_template('upload.html')


if __name__ == "__main__":
    app.run(debug=True)