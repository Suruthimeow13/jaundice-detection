from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = load_model('unet_jaundice_classifier.keras')

def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None

    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            path = os.path.join('static/uploads', image_file.filename)
            image_file.save(path)

            img = preprocess_image(path)
            pred = model.predict(img)[0][0]
            prediction = "Normal" if pred >= 0.5 else "Jaundice"
            confidence = round(float(pred if pred >= 0.5 else 1 - pred), 2)

            return render_template('index.html', prediction=prediction, confidence=confidence, image_path=path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)