from flask import Flask, request, jsonify, redirect
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warnings

import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load models
simple_cnn_model = load_model('Simple_Cnn.h5', compile=False)
mobilenet_model = load_model('Mobilenetvmodel.h5', compile=False)
vgg16_model = load_model('Vgg16.h5', compile=False)
inception_model = load_model('InceptionV3.h5', compile=False)
mixdsv_model = load_model('MIXDSV.h5', compile=False)
vmi_model = load_model('VMI.h5', compile=False)
sdx_model = load_model('SDX.h5', compile=False)
sv_model = load_model('SV.h5', compile=False)
mi_model = load_model('MI.h5', compile=False)
dx_model = load_model('DX.h5', compile=False)
densenet_model = load_model('denseNet.h5', compile=False)
xception_model = load_model('xception.h5', compile=False)

# Dictionary to store models
models = {
    'simple_cnn': simple_cnn_model,
    'mobilenet': mobilenet_model,
    'vgg16': vgg16_model,
    'inception': inception_model,
    'mixdsv': mixdsv_model,
    'vmi': vmi_model,
    'sdx': sdx_model,
    'sv': sv_model,
    'mi': mi_model,
    'dx': dx_model,
    'densenet': densenet_model,
    'xception': xception_model
}

# Updated root route to redirect to /predict
@app.route('/')
def home():
    return redirect('/predict')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Malaria Detection</title>
    <style>
        .container {
            text-align: center;
            margin-top: 50px;
        }
        img {
            max-width: 100%;
            height: auto;
            margin-top: 20px;
        }
        .result {
            margin-top: 20px;
            font-size: 1.2em;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Malaria Detection System</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <label for="modelSelect">Choose a model:</label>
            <select id="modelSelect" name="model" required>
                <option value="mixdsv">MIXDSV.h5</option>
                <option value="vmi">VMI.h5</option>
                <option value="sdx">SDX.h5</option>
                <option value="sv">SV.h5</option>
                <option value="mi">MI.h5</option>
                <option value="dx">DX.h5</option>
                <option value="densenet">Densent.h5</option>
                <option value="simple_cnn">Simple_Cnn.h5</option>
                <option value="xception">Xception.h5</option>
                <option value="vgg16">VGG16</option>
                <option value="mobilenet">MobileNet</option>
                <option value="inception">Inception</option>
            </select>
            <br><br>
            <input type="file" id="fileInput" name="file" accept="image/*" required>
            <br><br>
            <button type="button" onclick="submitForm()">Upload and Predict</button>
        </form>
        <div id="imageContainer"></div>
        <p id="result" class="result"></p>
    </div>

    <script>
        function submitForm() {
            var formData = new FormData();
            var modelName = document.getElementById('modelSelect').value;
            var fileInput = document.getElementById('fileInput');
            var file = fileInput.files[0];
            formData.append('file', file);
            formData.append('model', modelName);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('result').innerText = 'Error: ' + data.error;
                } else {
                    document.getElementById('result').innerText = 'Prediction using ' + modelName + ': ' + data.prediction + ' (Confidence: ' + data.confidence + ')';
                    displayImage(file);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('result').innerText = 'An error occurred. Please try again.';
            });
        }

        function displayImage(file) {
            var reader = new FileReader();
            reader.onload = function(event) {
                var imageContainer = document.getElementById('imageContainer');
                imageContainer.innerHTML = '<img src="' + event.target.result + '" alt="Uploaded Image">';
            }
            reader.readAsDataURL(file);
        }
    </script>
</body>
</html>
'''

    try:
        # Get the model name from the request
        model_name = request.form.get('model')
        if model_name not in models:
            return jsonify({'error': 'Invalid model name. Available models: simple_cnn, mobilenet, vgg16, inception, mixdsv, vmi, sdx, sv, mi, dx, densenet, xception'}), 400

        # Get the image file from the request
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided'}), 400

        # Preprocess the image
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224))  # Resize as per the model's requirement
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the chosen model
        model = models[model_name]
        predictions = model.predict(img_array)

        # Since the model uses sigmoid activation, it outputs a probability between 0 and 1
        probability = predictions[0][0]  # Extract the predicted probability

        # Interpret the prediction based on the probability value
        if probability > 0.5:
            prediction_label = 'Uninfected'
            confidence = probability * 100
        else:
            prediction_label = 'Parasitized'
            confidence = (1 - probability) * 100

        return jsonify({
            'model': model_name,
            'prediction': prediction_label,
            'confidence': f'{confidence:.2f}%'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New test route for verifying POST requests
@app.route('/test', methods=['POST'])
def test():
    return jsonify({'message': 'POST request received successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5004)
