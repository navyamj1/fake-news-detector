from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    vect_text = vectorizer.transform([text])
    prediction = model.predict(vect_text)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=False)
