import numpy as np
from flask import Flask, request, jsonify, render_template
from data_preprocess import*
import pickle


app = Flask(__name__)
model = pickle.load(open('finalized_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    print(int_features)
    heatId = int_features[0]
    mat_list = int_features[1]
    try:
        final_features = testing_data(heatId, mat_list, recp)   #making the input feature according to the feed of model
        prediction = model.predict(final_features)
        print(prediction)
        output = get_result(prediction[0], heatId, temp_data)
        print(output)
        return render_template('index.html', prediction_text='Defect type for the given input is {}'.format(output))
    except:
        return render_template('index.html', prediction_text='Error')
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)