// Get video canvas and alert box elements
const videoCanvas = document.getElementById('videoCanvas');
const alertBox = document.getElementById('alertBox');

// Function to display alerts
function displayAlert(message, color) {
    alertBox.textContent = message;
    alertBox.style.backgroundColor = color;
    alertBox.style.display = 'block';
    setTimeout(() => {
        alertBox.style.display = 'none';
    }, 3000);  // Hide after 3 seconds
}

// Function to draw webcam feed on canvas and send image data to backend
function drawFrame(video, context, width, height) {
    context.drawImage(video, 0, 0, width, height);
    const imageData = context.getImageData(0, 0, width, height);
    const formData = new FormData();
    formData.append('image', imageData);

    fetch('/process_image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Handle alert from backend
        if (data.alert) {
            displayAlert(data.alert === 'exam_terminated' ? 'ALERT: Malpractice detected!' : 'CAUTION: Cell Phone Detected!', data.alert === 'exam_terminated' ? '#ff0000' : '#ffff00');
        }
    })
    .catch(error => console.error('Error processing image:', error));

    requestAnimationFrame(() => drawFrame(video, context, width, height));
}

// Start webcam feed and draw frames on canvas
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        const video = document.createElement('video');
        video.srcObject = stream;
        video.play();
        const context = videoCanvas.getContext('2d');
        video.addEventListener('loadedmetadata', () => {
            videoCanvas.width = video.videoWidth;
            videoCanvas.height = video.videoHeight;
            drawFrame(video, context, videoCanvas.width, videoCanvas.height);
        });
    })
    .catch((error) => {
        console.error('Error accessing webcam:', error);
    });
