from flask import Flask, request, jsonify
from model import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        # Get data from POST request
        data = request.get_json()
        
        # Validate input
        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
        
        features = data['features']
        
        # Make prediction
        prediction = predict(features)
        
        # Return prediction
        return jsonify({'prediction': int(prediction)})
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0')