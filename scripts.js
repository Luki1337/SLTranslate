document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('uploadImageInput').addEventListener('change', async function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = async function(e) {
            const uploadedImage = document.getElementById('uploadedImage');
            uploadedImage.src = e.target.result;
            // Send the image file to the backend

            const prediction = await sendImageToBackend(file);
            debugger
            };
            reader.readAsDataURL(file);
        }
    });
});

document.getElementById('settingsBtn').addEventListener('click', function() {
    const settingsSection = document.getElementById('settings');
    settingsSection.style.display = settingsSection.style.display === 'none' ? 'block' : 'none';
});

document.getElementById('cameraBtn').addEventListener('click', function() {
    const cameraSection = document.getElementById('camera');
    cameraSection.style.display = cameraSection.style.display === 'none' ? 'block' : 'none';
    startCamera();
});

document.getElementById('applySettingsBtn').addEventListener('click', function() {
    const bgColor = document.getElementById('bgColor').value;
    const textColor = document.getElementById('textColor').value;
    const fontSize = document.getElementById('fontSize').value;
    const width = document.getElementById('width').value;
    const height = document.getElementById('height').value;

    document.body.style.backgroundColor = bgColor;
    document.body.style.color = textColor;
    document.body.style.fontSize = fontSize + 'px';
    document.body.style.width = width + '%';
    document.body.style.height = height + '%';
});

document.getElementById('deleteImageBtn').addEventListener('click', function() {
    const uploadedImage = document.getElementById('uploadedImage');
    uploadedImage.src = '';
    document.getElementById('uploaded-image').style.display = 'none';
    document.getElementById('uploadImageInput').value = ''; // Reset the file input

    // Reset the prediction section
    resetPredictionSection();
});

let aiResponseVariation = false;
document.getElementById('toggleResponseBtn').addEventListener('click', function() {
    aiResponseVariation = !aiResponseVariation;
    console.log('AI Response Variation ' + (aiResponseVariation ? 'Enabled' : 'Disabled'));
});

function startCamera() {
    const video = document.getElementById('video');

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
            video.srcObject = stream;
            video.play();
        });
    }
}

document.getElementById('captureBtn').addEventListener('click', function() {
    const canvas = document.getElementById('canvas');
    const video = document.getElementById('video');
    const context = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert the canvas to a blob and send it to the backend
    canvas.toBlob(async function(blob) {
        const file = new File([blob], 'captured.png', { type: 'image/png' });
        try {
            const prediction = await sendImageToBackend(file);
            displayPrediction(prediction);
        } catch (error) {
            console.error('Error sending image to backend:', error);
            // Handle error if necessary
        }
    }, 'image/png');
});

// Reset prediction section (hide image and clear prediction)
function resetPredictionSection() {
    const defaultImage = document.getElementById('defaultImage');
    defaultImage.src = '';
    defaultImage.style.display = 'none';

    const predictedLetter = document.getElementById('predictedLetter');
    predictedLetter.textContent = '';

    const predictionSection = document.getElementById('prediction');
    predictionSection.style.display = 'none';
}


async function sendImageToBackend(imageFile) {
    const formData = new FormData();
    formData.append('image', imageFile);
    console.debug("Sending image to backend");

    try {
        const response = await fetch('http://localhost:5000/classify-image', {
            method: 'POST',
            body: formData,
            mode: 'cors',
            headers: {
                'Accept': 'application/json'
            }
        });

        if (response.ok) {
            const data = await response.json();
            console.log(data);
            return data
        } else {
            const errorMessage = await response.text();
            console.error("Error:", errorMessage);
            throw new Error(errorMessage);
        }
    } catch (error) {
        console.error("Error:", error);
    }
}