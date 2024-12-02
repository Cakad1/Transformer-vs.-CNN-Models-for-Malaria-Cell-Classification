import os
import torch
import timm
import torch.nn as nn
from flask import Flask, request, render_template, jsonify
from PIL import Image
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    VIT_MODEL_PATH = os.path.join(BASE_DIR, 'vit_best.pth')
    DEIT_MODEL_PATH = os.path.join(BASE_DIR, 'deit_best.pth')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')
    IMG_SIZE = 224
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, 
           template_folder=Config.TEMPLATE_FOLDER)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER

# Create required directories
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.TEMPLATE_FOLDER, exist_ok=True)

def create_model(model_type):
    """Create model architecture based on type"""
    if model_type == "vit":
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
    elif model_type == "deit":
        model = timm.create_model('deit_base_patch16_224', pretrained=False, num_classes=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    n_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    return model

def load_models():
    """Load both ViT and DeiT models"""
    models = {}
    model_paths = {
        "vit": Config.VIT_MODEL_PATH,
        "deit": Config.DEIT_MODEL_PATH
    }
    
    for model_type, model_path in model_paths.items():
        if os.path.exists(model_path):
            try:
                model = create_model(model_type)
                model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
                model.to(Config.DEVICE)
                model.eval()
                models[model_type] = model
                print(f"{model_type.upper()} model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading {model_type.upper()} model: {str(e)}")
        else:
            print(f"{model_type.upper()} model not found at {model_path}")
    
    if not models:
        raise FileNotFoundError("No models found!")
    
    return models

def preprocess_image(image_path):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# Load models globally
print(f"Initializing models... Device: {Config.DEVICE}")
try:
    models = load_models()
except FileNotFoundError as e:
    print(f"Error initializing models: {e}")

@app.route('/', methods=['GET'])
def home():
    available_models = list(models.keys())
    return render_template('index.html', models=available_models)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    model_type = request.form.get('model', 'vit')  # Default to ViT if not specified
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if model_type not in models:
        return jsonify({'error': f'Model {model_type} not available'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess and predict
            image = preprocess_image(filepath)
            image = image.to(Config.DEVICE)

            with torch.no_grad():
                output = models[model_type](image)
                probability = output.item()  # Extract the predicted probability

            # Interpret the prediction based on the probability value
            if probability > 0.5:
                prediction_label = 'Parasitized'
                confidence = probability * 100
            else:
                prediction_label = 'Uninfected'
                confidence = (1 - probability) * 100
            
            # Clean up
            os.remove(filepath)

            return jsonify({
                'model_used': model_type,
                'prediction': prediction_label,
                'confidence': f'{confidence:.2f}%',
                'status': 'success'
            })
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500
            
    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    print(f"Starting Flask app...")
    print(f"Available models: {list(models.keys())}")
    print(f"Upload folder: {Config.UPLOAD_FOLDER}")
    print(f"Template folder: {Config.TEMPLATE_FOLDER}")
    app.run(debug=True, port=5002)
