// API Configuration - use same origin as current page
const API_BASE = window.location.origin;

// Global state
let image1Data = null;
let image2Data = null;
let detectImageData = null;
let embedImageData = null;
let videoStream = null;

// Check model status on load
window.addEventListener('load', () => {
    checkStatus();
});

// Check if model is loaded
async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        
        const indicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        
        if (data.model_loaded) {
            indicator.classList.add('online');
            statusText.textContent = `Model Ready (${data.device})`;
        } else {
            statusText.textContent = 'Model Loading...';
        }
    } catch (error) {
        console.error('Error checking status:', error);
        document.getElementById('statusText').textContent = 'Server Offline';
    }
}

// Tab switching
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + 'Tab').classList.add('active');
    event.target.closest('.tab-button').classList.add('active');
}

// File upload handlers
function handleFileUpload(personNum) {
    const file = document.getElementById(`file${personNum}`).files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById(`preview${personNum}`);
        preview.src = e.target.result;
        preview.style.display = 'block';
        
        const uploadArea = document.getElementById(`upload${personNum}`);
        uploadArea.querySelector('i').style.display = 'none';
        uploadArea.querySelector('p').style.display = 'none';
        
        if (personNum === 1) {
            image1Data = e.target.result;
        } else {
            image2Data = e.target.result;
        }
    };
    reader.readAsDataURL(file);
}

function handleDetectUpload() {
    const file = document.getElementById('fileDetect').files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('previewDetect');
        preview.src = e.target.result;
        preview.style.display = 'block';
        
        const uploadArea = document.getElementById('uploadDetect');
        uploadArea.querySelector('i').style.display = 'none';
        uploadArea.querySelector('p').style.display = 'none';
        
        detectImageData = e.target.result;
    };
    reader.readAsDataURL(file);
}

function handleEmbedUpload() {
    const file = document.getElementById('fileEmbed').files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('previewEmbed');
        preview.src = e.target.result;
        preview.style.display = 'block';
        
        const uploadArea = document.getElementById('uploadEmbed');
        uploadArea.querySelector('i').style.display = 'none';
        uploadArea.querySelector('p').style.display = 'none';
        
        embedImageData = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Verify faces
async function verifyFaces() {
    if (!image1Data || !image2Data) {
        alert('Please upload both images');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/api/verify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image1: image1Data,
                image2: image2Data
            })
        });
        
        const data = await response.json();
        showLoading(false);
        
        if (data.success) {
            displayVerifyResult(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        showLoading(false);
        alert('Error: ' + error.message);
    }
}

function displayVerifyResult(data) {
    const resultBox = document.getElementById('verifyResult');
    const content = document.getElementById('verifyContent');
    
    const badgeClass = data.is_same_person ? 'badge-success' : 'badge-danger';
    const badgeText = data.is_same_person ? '✓ Same Person' : '✗ Different Person';
    const badgeIcon = data.is_same_person ? 'fa-check-circle' : 'fa-times-circle';
    
    content.innerHTML = `
        <div class="result-item" style="justify-content: center; font-size: 1.2rem;">
            <span class="result-badge ${badgeClass}">
                <i class="fas ${badgeIcon}"></i> ${badgeText}
            </span>
        </div>
        
        <div class="result-item">
            <strong><i class="fas fa-percentage"></i> Confidence</strong>
            <span>${data.confidence.toFixed(2)}%</span>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" style="width: ${data.confidence}%">
                ${data.confidence.toFixed(1)}%
            </div>
        </div>
        
        <div class="result-item">
            <strong><i class="fas fa-ruler"></i> Cosine Similarity</strong>
            <span>${data.similarity.toFixed(4)}</span>
        </div>
        
        <div class="result-item">
            <strong><i class="fas fa-arrows-alt-h"></i> Cosine Distance</strong>
            <span>${data.distance.toFixed(4)}</span>
        </div>
        
        <div class="result-item">
            <strong><i class="fas fa-sliders-h"></i> Threshold</strong>
            <span>${data.threshold.toFixed(4)}</span>
        </div>
    `;
    
    resultBox.style.display = 'block';
}

// Detect faces
async function detectFaces() {
    if (!detectImageData) {
        alert('Please upload an image');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/api/detect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: detectImageData
            })
        });
        
        const data = await response.json();
        showLoading(false);
        
        if (data.success) {
            displayDetectResult(data);
        } else {
            alert('Message: ' + data.message);
        }
    } catch (error) {
        showLoading(false);
        alert('Error: ' + error.message);
    }
}

function displayDetectResult(data) {
    const resultBox = document.getElementById('detectResult');
    const content = document.getElementById('detectContent');
    
    let boxesHTML = '';
    if (data.boxes && data.boxes.length > 0) {
        boxesHTML = data.boxes.map((box, idx) => `
            <div class="result-item">
                <strong><i class="fas fa-user"></i> Face ${idx + 1}</strong>
                <span class="result-badge badge-info">${(data.confidences[idx] * 100).toFixed(1)}% confidence</span>
            </div>
        `).join('');
    }
    
    content.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${data.faces}</div>
            <div class="stat-label">Faces Detected</div>
        </div>
        
        ${boxesHTML}
    `;
    
    resultBox.style.display = 'block';
}

// Extract embedding
async function extractEmbedding() {
    if (!embedImageData) {
        alert('Please upload an image');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/api/embed`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: embedImageData
            })
        });
        
        const data = await response.json();
        showLoading(false);
        
        if (data.success) {
            displayEmbedResult(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        showLoading(false);
        alert('Error: ' + error.message);
    }
}

function displayEmbedResult(data) {
    const resultBox = document.getElementById('embedResult');
    const content = document.getElementById('embedContent');
    
    const embedding = data.embedding;
    const mean = embedding.reduce((a, b) => a + b, 0) / embedding.length;
    const variance = embedding.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / embedding.length;
    const std = Math.sqrt(variance);
    const norm = Math.sqrt(embedding.reduce((a, b) => a + b * b, 0));
    
    content.innerHTML = `
        <div class="embedding-stats">
            <div class="stat-card">
                <div class="stat-value">${data.dimension}</div>
                <div class="stat-label">Dimensions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${mean.toFixed(3)}</div>
                <div class="stat-label">Mean</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${std.toFixed(3)}</div>
                <div class="stat-label">Std Dev</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${norm.toFixed(3)}</div>
                <div class="stat-label">L2 Norm</div>
            </div>
        </div>
        
        <div class="embedding-box">
[${embedding.map(v => v.toFixed(6)).join(',\n ')}]
        </div>
    `;
    
    resultBox.style.display = 'block';
}

// Camera functions
async function startCamera() {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user' } 
        });
        
        const video = document.getElementById('video');
        video.srcObject = videoStream;
        
        document.getElementById('startCameraBtn').style.display = 'none';
        document.getElementById('stopCameraBtn').style.display = 'inline-flex';
        document.getElementById('captureBtn').style.display = 'inline-flex';
    } catch (error) {
        alert('Error accessing camera: ' + error.message);
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    
    document.getElementById('startCameraBtn').style.display = 'inline-flex';
    document.getElementById('stopCameraBtn').style.display = 'none';
    document.getElementById('captureBtn').style.display = 'none';
}

function capturePhoto() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg');
    
    const resultBox = document.getElementById('cameraResult');
    const content = document.getElementById('cameraContent');
    
    content.innerHTML = `
        <img src="${imageData}" style="max-width: 100%; border-radius: 10px; margin-bottom: 15px;">
        <button class="btn btn-primary" onclick="useCapturedPhoto('${imageData}')">
            <i class="fas fa-check"></i> Use This Photo
        </button>
    `;
    
    resultBox.style.display = 'block';
}

function useCapturedPhoto(imageData) {
    // Store captured photo globally
    const choice = confirm('Use this photo for:\nOK = Face Detection\nCancel = Face Verification (Person 1)');
    
    if (choice) {
        // Use for detection
        detectImageData = imageData;
        document.getElementById('previewDetect').src = imageData;
        document.getElementById('previewDetect').style.display = 'block';
        showTab('detect');
        alert('Photo loaded in Face Detection tab!');
    } else {
        // Use for verification as Person 1
        image1Data = imageData;
        document.getElementById('preview1').src = imageData;
        document.getElementById('preview1').style.display = 'block';
        showTab('verify');
        alert('Photo loaded as Person 1 in Face Verification tab!');
    }
}

// Loading overlay
function showLoading(show) {
    document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
}
