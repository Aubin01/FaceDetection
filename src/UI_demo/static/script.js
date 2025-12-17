// Author: Aubin Mugisha & Copilot
// Description: Front-end interactions for FaceID Pro demo (uploads, camera, API calls).
// =====================================================
// FaceID Pro - Modern Neural Interface
// =====================================================

const API_BASE = window.location.origin;

// Global state
let image1Data = null;
let image2Data = null;
let detectImageData = null;
let embedImageData = null;
let videoStream = null;
let videoStream2 = null;
let cameraPerson1Data = null;
let cameraPerson2Data = null;

// Page titles for each tab
const pageTitles = {
    verify: { title: 'Face Verification', description: 'Compare two faces to verify identity match' },
    detect: { title: 'Face Detection', description: 'Scan an image to detect and locate faces' },
    embed: { title: 'Embedding Extraction', description: 'Generate 128D biometric face vectors' },
    camera: { title: 'Live Capture', description: 'Real-time face capture and verification' }
};

// Initialize on load
window.addEventListener('load', () => {
    checkStatus();
    initDragDrop();
});

// Check model status
async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        
        const indicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        
        if (data.model_loaded) {
            indicator.classList.add('online');
            statusText.textContent = `Ready • ${data.device.toUpperCase()}`;
        } else {
            statusText.textContent = 'Initializing...';
        }
    } catch (error) {
        console.error('Status check failed:', error);
        document.getElementById('statusText').textContent = 'Offline';
    }
}

// Tab switching
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + 'Tab').classList.add('active');
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    
    // Update page header
    const pageInfo = pageTitles[tabName];
    document.getElementById('pageTitle').textContent = pageInfo.title;
    document.getElementById('pageDescription').textContent = pageInfo.description;
}

// Initialize drag & drop
function initDragDrop() {
    const zones = document.querySelectorAll('.upload-zone');
    
    zones.forEach(zone => {
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        });
        
        zone.addEventListener('dragleave', () => {
            zone.classList.remove('dragover');
        });
        
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                const zoneId = zone.id;
                if (zoneId === 'upload1') handleDroppedFile(file, 1);
                else if (zoneId === 'upload2') handleDroppedFile(file, 2);
                else if (zoneId === 'uploadDetect') handleDroppedFileDetect(file);
                else if (zoneId === 'uploadEmbed') handleDroppedFileEmbed(file);
            }
        });
    });
}

function handleDroppedFile(file, personNum) {
    const reader = new FileReader();
    reader.onload = (e) => {
        updateUploadPreview(personNum, e.target.result);
        if (personNum === 1) image1Data = e.target.result;
        else image2Data = e.target.result;
    };
    reader.readAsDataURL(file);
}

function handleDroppedFileDetect(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        updateDetectPreview(e.target.result);
        detectImageData = e.target.result;
    };
    reader.readAsDataURL(file);
}

function handleDroppedFileEmbed(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        updateEmbedPreview(e.target.result);
        embedImageData = e.target.result;
    };
    reader.readAsDataURL(file);
}

// File upload handlers
function handleFileUpload(personNum) {
    const file = document.getElementById(`file${personNum}`).files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        updateUploadPreview(personNum, e.target.result);
        if (personNum === 1) image1Data = e.target.result;
        else image2Data = e.target.result;
    };
    reader.readAsDataURL(file);
}

function updateUploadPreview(personNum, imageData) {
    const preview = document.getElementById(`preview${personNum}`);
    const zone = document.getElementById(`upload${personNum}`);
    const content = zone.querySelector('.upload-content');
    
    preview.src = imageData;
    preview.style.display = 'block';
    content.style.display = 'none';
    zone.classList.add('has-image');
}

function handleDetectUpload() {
    const file = document.getElementById('fileDetect').files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        updateDetectPreview(e.target.result);
        detectImageData = e.target.result;
    };
    reader.readAsDataURL(file);
}

function updateDetectPreview(imageData) {
    const preview = document.getElementById('previewDetect');
    const zone = document.getElementById('uploadDetect');
    const content = zone.querySelector('.upload-content');
    
    preview.src = imageData;
    preview.style.display = 'block';
    content.style.display = 'none';
    zone.classList.add('has-image');
}

function handleEmbedUpload() {
    const file = document.getElementById('fileEmbed').files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        updateEmbedPreview(e.target.result);
        embedImageData = e.target.result;
    };
    reader.readAsDataURL(file);
}

function updateEmbedPreview(imageData) {
    const preview = document.getElementById('previewEmbed');
    const zone = document.getElementById('uploadEmbed');
    const content = zone.querySelector('.upload-content');
    
    preview.src = imageData;
    preview.style.display = 'block';
    content.style.display = 'none';
    zone.classList.add('has-image');
}

// =====================================================
// Face Verification
// =====================================================
async function verifyFaces() {
    if (!image1Data || !image2Data) {
        showNotification('Please upload both images', 'warning');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/api/verify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image1: image1Data, image2: image2Data })
        });
        
        const data = await response.json();
        showLoading(false);
        
        if (data.success) {
            displayVerifyResult(data);
        } else {
            showNotification('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showLoading(false);
        showNotification('Error: ' + error.message, 'error');
    }
}

function displayVerifyResult(data) {
    const resultBox = document.getElementById('verifyResult');
    const content = document.getElementById('verifyContent');
    
    const isMatch = data.is_same_person;
    const iconClass = isMatch ? 'success' : 'danger';
    const icon = isMatch ? 'fa-check' : 'fa-xmark';
    const title = isMatch ? 'Same Person ✓' : 'Different People';
    const subtitle = isMatch 
        ? 'These faces belong to the same person' 
        : 'These faces belong to different people';
    
    // Use actual similarity score (0-100%)
    const similarityPercent = (data.similarity * 100).toFixed(1);
    const matchLevel = similarityPercent >= 80 ? 'Very High' : 
                       similarityPercent >= 60 ? 'High' : 
                       similarityPercent >= 40 ? 'Medium' : 
                       similarityPercent >= 20 ? 'Low' : 'Very Low';
    
    // For the bar, clamp between 0-100
    const barWidth = Math.max(0, Math.min(100, similarityPercent));
    
    content.innerHTML = `
        <div class="match-result">
            <div class="match-icon ${iconClass}">
                <i class="fas ${icon}"></i>
            </div>
            <h2 class="match-title ${iconClass}">${title}</h2>
            <p class="match-subtitle">${subtitle}</p>
            
            <div class="match-score-display">
                <div class="score-circle ${iconClass}">
                    <span class="score-number">${Math.round(similarityPercent)}</span>
                    <span class="score-percent">%</span>
                </div>
                <div class="score-info">
                    <span class="score-label">Similarity Score</span>
                    <span class="score-level ${iconClass}">${matchLevel}</span>
                </div>
            </div>
            
            <div class="match-meter">
                <div class="meter-labels">
                    <span>Different</span>
                    <span>Same Person</span>
                </div>
                <div class="meter-track">
                    <div class="meter-fill ${iconClass}" style="width: ${barWidth}%"></div>
                </div>
                <div class="threshold-info">
                    <span>Distance: ${data.distance.toFixed(4)}</span>
                    <span>Threshold: ${data.threshold.toFixed(4)}</span>
                </div>
            </div>
        </div>
    `;
    
    resultBox.style.display = 'block';
    resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// =====================================================
// Face Detection
// =====================================================
async function detectFaces() {
    if (!detectImageData) {
        showNotification('Please upload an image', 'warning');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/api/detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: detectImageData })
        });
        
        const data = await response.json();
        showLoading(false);
        
        if (data.success) {
            displayDetectResult(data);
        } else {
            showNotification(data.message || 'No faces detected', 'info');
        }
    } catch (error) {
        showLoading(false);
        showNotification('Error: ' + error.message, 'error');
    }
}

function displayDetectResult(data) {
    const resultBox = document.getElementById('detectResult');
    const content = document.getElementById('detectContent');
    
    // Update the preview image with bounding boxes
    if (data.image_with_boxes) {
        const preview = document.getElementById('previewDetect');
        preview.src = data.image_with_boxes;
    }
    
    let facesHTML = '';
    if (data.boxes && data.boxes.length > 0) {
        facesHTML = data.boxes.map((box, idx) => `
            <div class="face-detected">
                <div class="face-number">${idx + 1}</div>
                <div class="face-info">
                    <h4>Face ${idx + 1}</h4>
                    <p>Confidence: ${(data.confidences[idx] * 100).toFixed(1)}%</p>
                </div>
            </div>
        `).join('');
    }
    
    content.innerHTML = `
        <div class="match-result">
            <div class="match-icon success">
                <i class="fas fa-face-smile"></i>
            </div>
            <h2 class="match-title success">${data.faces} Face${data.faces !== 1 ? 's' : ''} Detected</h2>
            <p class="match-subtitle">Successfully identified faces in the image</p>
        </div>
        
        <div class="detection-results">
            ${facesHTML}
        </div>
    `;
    
    resultBox.style.display = 'block';
}

// =====================================================
// Embedding Extraction
// =====================================================
async function extractEmbedding() {
    if (!embedImageData) {
        showNotification('Please upload an image', 'warning');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/api/embed`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: embedImageData })
        });
        
        const data = await response.json();
        showLoading(false);
        
        if (data.success) {
            displayEmbedResult(data);
        } else {
            showNotification('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showLoading(false);
        showNotification('Error: ' + error.message, 'error');
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
    const min = Math.min(...embedding);
    const max = Math.max(...embedding);
    
    content.innerHTML = `
        <div class="match-result">
            <div class="match-icon success">
                <i class="fas fa-dna"></i>
            </div>
            <h2 class="match-title success">Embedding Generated</h2>
            <p class="match-subtitle">128-dimensional biometric signature</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value accent">${data.dimension}</div>
                <div class="stat-label">Dimensions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${mean.toFixed(4)}</div>
                <div class="stat-label">Mean</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${std.toFixed(4)}</div>
                <div class="stat-label">Std Dev</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success">${norm.toFixed(4)}</div>
                <div class="stat-label">L2 Norm</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${min.toFixed(4)}</div>
                <div class="stat-label">Min</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${max.toFixed(4)}</div>
                <div class="stat-label">Max</div>
            </div>
        </div>
        
        <div class="embedding-display">
[${embedding.map(v => v.toFixed(6)).join(', ')}]
        </div>
    `;
    
    resultBox.style.display = 'block';
}

// =====================================================
// Camera Functions
// =====================================================
async function startCamera() {
    try {
        cameraPerson1Data = null;
        cameraPerson2Data = null;
        
        videoStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user', width: 640, height: 480 } 
        });
        
        const video = document.getElementById('video');
        video.srcObject = videoStream;
        
        document.getElementById('cameraPreview1').style.display = 'block';
        document.getElementById('capturedPreview1').style.display = 'none';
        document.getElementById('rec1').style.display = 'flex';
        document.getElementById('startCameraBtn').style.display = 'none';
        document.getElementById('stopCameraBtn').style.display = 'flex';
        document.getElementById('captureBtn1').style.display = 'flex';
        document.getElementById('cameraVerifyResult').style.display = 'none';
    } catch (error) {
        showNotification('Camera access denied: ' + error.message, 'error');
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    if (videoStream2) {
        videoStream2.getTracks().forEach(track => track.stop());
        videoStream2 = null;
    }
    
    document.getElementById('cameraPreview1').style.display = 'block';
    document.getElementById('cameraPreview2').style.display = 'none';
    document.getElementById('capturedPreview1').style.display = 'none';
    document.getElementById('capturedPreview2').style.display = 'none';
    document.getElementById('waitingMessage').style.display = 'flex';
    document.getElementById('rec1').style.display = 'none';
    document.getElementById('rec2').style.display = 'none';
    
    document.getElementById('startCameraBtn').style.display = 'flex';
    document.getElementById('stopCameraBtn').style.display = 'none';
    document.getElementById('captureBtn1').style.display = 'none';
    document.getElementById('captureBtn2').style.display = 'none';
    document.getElementById('verifyCameraBtn').style.display = 'none';
    
    cameraPerson1Data = null;
    cameraPerson2Data = null;
}

async function capturePhoto1() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    
    cameraPerson1Data = canvas.toDataURL('image/jpeg', 0.9);
    
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    
    document.getElementById('capturedImg1').src = cameraPerson1Data;
    document.getElementById('cameraPreview1').style.display = 'none';
    document.getElementById('capturedPreview1').style.display = 'block';
    document.getElementById('rec1').style.display = 'none';
    document.getElementById('captureBtn1').style.display = 'none';
    
    // Start camera for Person 2
    try {
        videoStream2 = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user', width: 640, height: 480 } 
        });
        
        const video2 = document.getElementById('video2');
        video2.srcObject = videoStream2;
        
        document.getElementById('waitingMessage').style.display = 'none';
        document.getElementById('cameraPreview2').style.display = 'block';
        document.getElementById('rec2').style.display = 'flex';
        document.getElementById('captureBtn2').style.display = 'flex';
    } catch (error) {
        showNotification('Camera error: ' + error.message, 'error');
    }
}

function capturePhoto2() {
    const video = document.getElementById('video2');
    const canvas = document.getElementById('canvas2');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    
    cameraPerson2Data = canvas.toDataURL('image/jpeg', 0.9);
    
    if (videoStream2) {
        videoStream2.getTracks().forEach(track => track.stop());
        videoStream2 = null;
    }
    
    document.getElementById('capturedImg2').src = cameraPerson2Data;
    document.getElementById('cameraPreview2').style.display = 'none';
    document.getElementById('capturedPreview2').style.display = 'block';
    document.getElementById('rec2').style.display = 'none';
    document.getElementById('captureBtn2').style.display = 'none';
    document.getElementById('verifyCameraBtn').style.display = 'flex';
}

function retakePhoto(person) {
    if (person === 1) {
        cameraPerson1Data = null;
        cameraPerson2Data = null;
        document.getElementById('capturedPreview1').style.display = 'none';
        document.getElementById('capturedPreview2').style.display = 'none';
        document.getElementById('cameraPreview2').style.display = 'none';
        document.getElementById('waitingMessage').style.display = 'flex';
        document.getElementById('verifyCameraBtn').style.display = 'none';
        document.getElementById('rec2').style.display = 'none';
        
        if (videoStream2) {
            videoStream2.getTracks().forEach(track => track.stop());
            videoStream2 = null;
        }
        
        startCamera();
    } else {
        cameraPerson2Data = null;
        document.getElementById('capturedPreview2').style.display = 'none';
        document.getElementById('verifyCameraBtn').style.display = 'none';
        
        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
            .then(stream => {
                videoStream2 = stream;
                document.getElementById('video2').srcObject = stream;
                document.getElementById('cameraPreview2').style.display = 'block';
                document.getElementById('rec2').style.display = 'flex';
                document.getElementById('captureBtn2').style.display = 'flex';
            })
            .catch(error => showNotification('Camera error: ' + error.message, 'error'));
    }
}

async function verifyCameraPhotos() {
    if (!cameraPerson1Data || !cameraPerson2Data) {
        showNotification('Please capture both photos', 'warning');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/api/verify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image1: cameraPerson1Data, image2: cameraPerson2Data })
        });
        
        const data = await response.json();
        showLoading(false);
        
        if (data.success) {
            displayCameraVerifyResult(data);
        } else {
            showNotification('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showLoading(false);
        showNotification('Error: ' + error.message, 'error');
    }
}

function displayCameraVerifyResult(data) {
    const resultBox = document.getElementById('cameraVerifyResult');
    const content = document.getElementById('cameraVerifyContent');
    
    const isMatch = data.is_same_person;
    const iconClass = isMatch ? 'success' : 'danger';
    const icon = isMatch ? 'fa-check' : 'fa-xmark';
    const title = isMatch ? 'Identity Match' : 'No Match';
    const subtitle = isMatch ? 'These appear to be the same person' : 'These appear to be different people';
    const similarityPercent = (data.similarity * 100).toFixed(1);
    
    content.innerHTML = `
        <div class="match-result">
            <div class="match-icon ${iconClass}">
                <i class="fas ${icon}"></i>
            </div>
            <h2 class="match-title ${iconClass}">${title}</h2>
            <p class="match-subtitle">${subtitle}</p>
            
            <div class="confidence-bar">
                <div class="confidence-track">
                    <div class="confidence-fill" style="width: ${similarityPercent}%"></div>
                </div>
                <div class="confidence-labels">
                    <span>0%</span>
                    <span>Similarity: ${similarityPercent}%</span>
                    <span>100%</span>
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value accent">${data.similarity.toFixed(4)}</div>
                    <div class="stat-label">Similarity</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.distance.toFixed(4)}</div>
                    <div class="stat-label">Distance</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.threshold.toFixed(4)}</div>
                    <div class="stat-label">Threshold</div>
                </div>
            </div>
        </div>
    `;
    
    resultBox.style.display = 'block';
    resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// =====================================================
// Utility Functions
// =====================================================
function showLoading(show) {
    document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
}

function showNotification(message, type = 'info') {
    // Simple alert for now - could be enhanced with toast notifications
    alert(message);
}
