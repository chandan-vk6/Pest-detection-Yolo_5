from flask import Flask, render_template, request
import os 
from deeplearning import object_detection
# webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')


@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        no_detection, label =  object_detection(path_save,filename)
        
        if label == '':
            label= 'Not Detected'

        return render_template('index.html',upload=True,upload_image=filename,text=label,no=no_detection)

    return render_template('index.html',upload=False)


if __name__ =="__main__":
    app.run(debug=True)