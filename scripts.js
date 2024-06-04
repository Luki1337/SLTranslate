document.getElementById('uploadImageInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const uploadedImage = document.getElementById('uploadedImage');
            uploadedImage.src = e.target.result;
            document.getElementById('uploaded-image').style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
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
});

let aiResponseVariation = false;
document.getElementById('toggleResponseBtn').addEventListener('click', function() {
    aiResponseVariation = !aiResponseVariation;
    alert('AI Response Variation ' + (aiResponseVariation ? 'Enabled' : 'Disabled'));
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

    // For demonstration, we'll display the captured image in an alert
    const imageDataUrl = canvas.toDataURL('image/png');
    alert('Image captured: ' + imageDataUrl);
});
