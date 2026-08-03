from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained SVC/ML model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        cgpa = float(data['cgpa'])
        iq = float(data['iq'])
        profile_score = float(data['profile_score'])

        # Input array matching training shape (1, 3)
        input_data = np.array([[cgpa, iq, profile_score]])
        
        # Predict outcome
        prediction = model.predict(input_data)[0]

        # Calculate feature benchmark percentages for UI gauge charts
        cgpa_pct = round((cgpa / 10.0) * 100, 1)
        iq_pct = round(((iq - 50) / 150) * 100, 1)  # scaled 50-200 IQ range
        profile_pct = round((profile_score / 100.0) * 100, 1)

        return jsonify({
            'success': True,
            'placed': int(prediction),
            'status': "High Chance of Placement! 🎉" if prediction == 1 else "Needs Improvement ⚠️",
            'scores': {
                'cgpa': cgpa,
                'iq': iq,
                'profile': profile_score
            },
            'percentages': {
                'cgpa': cgpa_pct,
                'iq': iq_pct,
                'profile': profile_pct
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)