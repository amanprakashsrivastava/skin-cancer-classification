import os
from flask import Flask, request, render_template, redirect, url_for,jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import torch
import torchvision.models as models
from utils import load_image, classify_image, detect_lesion

# Initialize Flask app
app = Flask(__name__)

# Set up paths and allowed extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'tif', 'webp', 'heif', 'heic'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the classification model (assuming it's a PyTorch model)
classification_model = torch.load('models/model.pth', map_location='cpu')  # Load the entire model
classification_model.eval()

# Load the lesion detection model (assuming it's a Keras model)
lesion_model = load_model('models/results.keras')

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')  # Your main page with the upload form

# @app.route('/upload', methods=['POST'])
# def upload_file():
    # if 'file' not in request.files:
    #     return redirect("hello",request.url)  # No file part

    # file = request.files['file']
    
    # # If no file is selected
    # if file.filename == '':
    #     return redirect(request.url)

    # # If the file is allowed, save it
    # if file and allowed_file(file.filename):
    #     filename = secure_filename(file.filename)
    #     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #     file.save(filepath)

    #     # Process the image
    #     image = load_image(filepath)
    #     classification_result = classify_image(classification_model, image)
    #     lesion_result = detect_lesion(lesion_model, image)

    #     # Now directly render the result page
    #     return render_template(
    #         'result.html', 
    #         classification=classification_result,
    #         lesion=lesion_result,
    #         image_url=filepath
    #     )

    # return redirect(request.url)  # If the file type is not allowed, redirect to the upload page
# @app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST','GET'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file to a local path
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # Classify the image
    classification_result = classify_image(classification_model, img_path)

    # Detect lesion and save the output image
    lesion_output_path = detect_lesion(lesion_model, load_image(img_path))

    # Render the results
    return render_template(
        'results.html',
        classification=classification_result,
        lesion_image_url=url_for('static', filename=os.path.basename(lesion_output_path)),
        original_image_url=url_for('static', filename=file.filename)
    )


# Ensure the app runs
if __name__ == '__main__':
    app.run(debug=True)
