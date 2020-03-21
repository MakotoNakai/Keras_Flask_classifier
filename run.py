from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

import keras.backend.tensorflow_backend as tb
import tensorflow.python.keras.backend as K
import numpy as np
import os, sys
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tb._SYMBOLIC_SCOPE.value = True

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

HAAR_FILE = "./tools/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(HAAR_FILE)
size = 50

model = load_model("./tools/ladies_classification.h5", compile=False)
# graph = K.get_graph()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return redirect(url_for('predict'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file.')
            return redirect(url_for('predict'))
            
        file = request.files['file']
        if file.filename == '':
            flash('No file.')
            return redirect(url_for('predict'))
            
        if file and is_allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            img = cv2.imread(filepath)
            
            # グレースケールに変換する?
            img_g = cv2.imread(filepath,0)
            
            # カスケード型分類器を使用して画像ファイルから顔部分を検出する
            face = cascade.detectMultiScale(img_g)
  
            # 顔の座標を表示する
            if len(face) != 0:

                for x,y,w,h in face:
                    face_cut = img[y:y+h, x:x+w]
        
                face_cut_resized = cv2.resize(face_cut, (size,  size))
                img_nad = img_to_array(face_cut_resized)/255
                img_nad = img_nad[None, ...]

#                 with graph.as_default():
                label=["is", "is not"]
                pred = model.predict(img_nad, batch_size=1, verbose=0)
                pred_label = label[np.argmax(pred[0])]
                result = "She {} Makoto Nakai's type".format(pred_label)
        
            else:
                result = "Face detection failed. Please try another image"
            
            return render_template('result.html', result=result, filepath=filepath)
    return render_template('predict.html')
    

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(threaded=False, debug=True)