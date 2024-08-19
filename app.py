import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import tensorflow as tf


app = Flask(__name__)


# Load the pre-trained ML model (replace 'model.h5' with your model file)
model = tf.keras.models.load_model('SIH_new_model.h5')

# Define class names (replace with your class labels)
class_names = ['Nitrogen Deficient', 'Phosphorous Deficient', 'Potassium Deficient']

# Configure a folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # Load and preprocess the image
        image = Image.open(image_path)
        image = image.resize((224, 224))
        actual_image=image  # Adjust the size as needed
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make a prediction using the ML model
        prediction = model.predict(image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = round(100 * np.max(prediction), 2)

        # Generate a unique result image filename
        # result_image_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')

        # Save the result image (modify as needed based on your requirements)
        # result_image = Image.new('RGB', (224, 224), (255, 255, 255))
        # result_image.save(result_image_filename)

        if predicted_class is 'Nitrogen Deficient':

            theory='''*Applying nitrogen fertilizer efficiently.\n
                    *Using available organic materials, such as farmyard manure, crop residues, or compost.\n
                    *Placing manure or fertilizer directly into the soil.\n
                    *Applying nitrogen fertilizer pre-flood and pre-plant in water-seeded rice.\n
                    *Controlling weeds that compete with rice for nitrogen.\n
                    *Consulting a local licensed agronomist.\n'''
            
        elif predicted_class is 'Phosphorous Deficient':
            theory='''*Applying rock phosphate 2-3 weeks before irrigation.\n
                    *Applying bone meal or manure.\n
                    *Applying phosphorous-rich fertilizer around the base of the plant.\n
                    *Applying a phosphoric acid fertilizer.\n
                    *Incorporating rice straw.\n'''
        
        elif predicted_class is 'Potassium Deficient':
            theory='''*Spread organic mulch beneath plants.\n
                    *Apply potassium fertilizer, preferably slow-release forms.\n
                    *Foliar spray of 1% K2SO4 or 1% KCl (10g/lit) at 15 days interval up to the disappearance of symptoms.\n
                    *Split K in at least two doses if soil is sandy with leaching.\n'''

        return render_template('result.html', theory=theory,predicted_class=predicted_class, confidence=confidence)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

