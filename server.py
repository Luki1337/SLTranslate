
from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import torch
from torchvision import transforms
import numpy as np
import network
app = Flask(__name__)

# Ensure the images directory exists
IMAGE_FOLDER = 'images'
DEFAULT_IMAGE_FOLDER = 'default_images'

if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Define the required parameters for normalization
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]
resize_size = 256
crop_size = 224


# Define custom transformation function
def custom_transform(image):
    # Convert numpy array to PIL Image
    image = np.array(image)
    image = Image.fromarray(np.uint8(image.squeeze()))
    return image


val_transform = transforms.Compose([
    custom_transform,
    transforms.Resize(resize_size, interpolation=Image.BICUBIC),
    transforms.CenterCrop(crop_size),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean, std=normalize_std)
])

# Load your PyTorch model (replace with your model)
model, _ = network.create_network(freeze_pretrained=False)
model.load_state_dict(torch.load("models/model.pth"))
model.eval()

# Create a dictionary to map model outputs to letters
output_to_letter = {i: chr(65 + i) for i in range(26)}


@app.route('/classify-image', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))

    # Save the image locally
    image_path = os.path.join(IMAGE_FOLDER, image_file.filename)
    image.save(image_path)

    # Load the image and apply transformations
    image = Image.open(image_path)
    image = val_transform(image).unsqueeze(0)  # Add batch dimension

    # Make a prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction_idx = predicted.item()
        predicted_letter = output_to_letter[prediction_idx]

    # Load the corresponding default image
    default_image_path = os.path.join(DEFAULT_IMAGE_FOLDER, f'{predicted_letter}.png')

    # Ensure the default image exists
    if not os.path.exists(default_image_path):
        return jsonify({'error': 'Default image not found'}), 404

    # Return the classification and the default image
    print(predicted_letter)
    return jsonify({
        'classification': predicted_letter
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True)
