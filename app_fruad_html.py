import pandas as pd
from flask import Flask, request, render_template
import joblib
import sys

app = Flask(__name__)

@app.route("/")
def home():
    # Define the columns expected by your pipeline
    cols =['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
           'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
           'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
    
    return render_template("index.html", cols=cols)

@app.route("/result", methods=["POST"])
def result():
    n_features = []
    
    for i in request.form.values():
        n_features.append(i)
    print(n_features)   
    
    # Define the columns expected by your pipeline
    cols =['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
           'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
           'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
    
    features = pd.DataFrame([n_features], columns=cols)
    trans_data = pipeline.transform(features)
    prediction = lr.predict(trans_data)
    
    return render_template("result.html", prediction=prediction[0])

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load('voting_clf.pkl') # Load "voting_clf.pkl"
    print('Model loaded')
    pipeline = joblib.load('pipeline.pkl')
    print('Pipeline loaded')
    
    app.run(port=port, debug=True)
