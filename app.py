#import Flask
import numpy as np
import joblib
from flask import Flask, render_template,request
#create an instance of Flask
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/',methods=['GET','POST'])
def predict():
    if request.method == "POST":
        sepal_length = request.form.get('sepal_length')
        sepal_width = request.form.get('sepal_width')
        petal_length = request.form.get('petal_length')
        petal_width = request.form.get('petal_width')
        
        try:
            prediction = preprocessDataAndPredict(sepal_length,sepal_width,petal_length,petal_width)
            
            return render_template('predict.html',prediction=prediction)
        
        except ValueError:
            return "Invalid Values. Try Again"
    pass
    pass

def predict_species_name(*result):
    predict_species_name= []
    for i in result:
        if (i==0 ):
            predict_species_name.append("Iris-setosa")
        elif (i==1):
            predict_species_name.append("Iris-versicolor")
        else:
            predict_species_name.append("Iris-virginica")
    
    return predict_species_name

def preprocessDataAndPredict(sepal_length,sepal_width,petal_length,petal_width):
    test_data = [sepal_length,sepal_width,petal_length,petal_width]
    
    print(test_data)
    
    test_data = np.array(test_data)
    test_data = test_data.reshape(1,-1)
    print(test_data)
    
    file = open(r"C:/Users/Dhakshesh/Documents/ML/Jupyter files/finalized_model.pkl","rb")
    trained_model = joblib.load(file)
    prediction = trained_model.predict(test_data)
    
    prediction = predict_species_name(*prediction)
    return prediction 
pass

if __name__ == '__main__':
    app.run(debug=False)