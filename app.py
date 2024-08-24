from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

def predict(w, b, x):
    return w * x + b

def gradient_descent(x, y, w, b, learning_rate, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        dw = (1/m) * np.sum((predict(w, b, x) - y) * x)
        db = (1/m) * np.sum(predict(w, b, x) - y)
        w -= learning_rate * dw
        b -= learning_rate * db
    return w, b

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    data = request.get_json()
    x = np.array(data['x'])
    y = np.array(data['y'])
    
    w_init = 0
    b_init = 0
    learning_rate = 0.0000001
    num_iterations = 10000
    w, b = gradient_descent(x, y, w_init, b_init, learning_rate, num_iterations)
    
    return jsonify({'w': w, 'b': b})

@app.route('/predict', methods=['POST'])
def predict_price():
    data = request.get_json()
    square_footage = float(data['square_footage'])
    w = float(data['w'])
    b = float(data['b'])
    predicted_price = predict(w, b, square_footage)
    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)


