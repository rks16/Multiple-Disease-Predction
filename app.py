from flask import Flask, render_template, request
import pickle
import numpy as np
from diabetes import scaler
from liver import scale

model= pickle.load(open('mdl.pkl','rb'))
model_hrt= pickle.load(open('hrt.pkl','rb'))
model_lvr = pickle.load(open('lvr.pkl','rb'))


app = Flask(__name__)


#Home Page render
@app.route('/')
def main():
    return render_template('index.html')

#Result page for Diabates render
@app.route('/result', methods=['POST'])
def pd():
    arr =[]
    
    for data in request.form.values():
        arr.append(float(data))
    arr = np.array(arr).reshape(1, -1)
    std_data = scaler.transform(arr)
    pred = model.predict(std_data)
    return render_template('result.html', data=pred)   


#Result page for Heart render
@app.route('/result-hrt', methods=['POST'])
def pd2():
    arr =[]
  
    for data in request.form.values():
        arr.append(float(data))
    arr = np.array(arr).reshape(1, -1)
  
    pred = model_hrt.predict(arr)
    return render_template('result-hrt.html', data=pred)  


#Result page for liver render
@app.route('/result-liver', methods=['POST'])
def pd3():
    arr =[]
    
    for data in request.form.values():
        arr.append(float(data))
    arr = np.array(arr).reshape(1, -1)
    std_data = scale.transform(arr)
    pred = model_lvr.predict(std_data)
    return render_template('result-liver.html', data=pred) 
   


if __name__ == "__main__":
    app.run(debug=True)