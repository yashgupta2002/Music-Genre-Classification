from flask import Flask,render_template, request,jsonify
import datetime
import os
from predict import predict
import json
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload",methods=["POST","GET"])
def upload():
    print(request.method)
    up_file=request.files["file"]
    
    d=str(datetime.datetime.now().timestamp())+".wav"
    
    file_path=os.path.join(app.root_path,"uploaded_files",d)
    up_file.save(file_path)
    
    predition=predict(file_path)
   
    return jsonify(predition).json


if __name__=='__main__':
    app.run(debug=True)