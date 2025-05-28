// Global variables
let currentImageData = null;
let analysisInProgress = false;

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    console.log('🌞 Solar Analyzer initialized');
    
    // Set default values
    const addressInput = document.getElementById('address');
    const electricityRateInput = document.getElementById('electricityRate');
    
    // Load saved values from localStorage
    if (localStorage.getItem('solarAnalyzer_address')) {
        addressInput.value = localStorage.getItem('solarAnalyzer_address');
    }
    
    if (localStorage.getItem('solarAnalyzer_rate')) {
        electricityRateInput.value = localStorage.getItem('solarAnalyzer_rate');
    }
    
    // Save values on change
    addressInput.addEventListener('change', () => {
        localStorage.setItem('solarAnalyzer_address', addressInput.value);
    });
    
    electricityRateInput.addEventListener('change', () => {
        localStorage.setItem('solarAnalyzer_rate', electricityRateInput.value);
    });
}

// Tab Management
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Clear preview and data
    clearImagePreview();
}

function clearImagePreview() {
    const preview = document.getElementById('imagePreview');
    const details = document.getElementById('imageAnalysisDetails');
    
    preview.style.display = 'none';
    details.style.display = 'none';
    currentImageData = null;
}

// Image Preview Functions
function previewImageFromUrl() {
    const url = document.getElementById('imageUrl').value;
    
    if (!url) {
        clearImagePreview();
        return;
    }
    
    // Validate URL format
    if (!isValidImageUrl(url)) {
        showError('Please enter a valid image URL');
        return;
    }
    
    const img = document.getElementById('imagePreview');
    const details = document.getElementById('imageAnalysisDetails');
    
    // Show loading state
    img.style.display = 'block';
    img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkxvYWRpbmcuLi48L3RleHQ+PC9zdmc+';
    
    // Test image loading
    const testImg = new Image();
    testImg.onload = function() {
        img.src = url;
        currentImageData = { type: 'url', data: url };
        
        // Show analysis details
        document.getElementById('imageSource').textContent = `Source: ${getUrlDomain(url)}`;
        details.style.display = 'block';
        
        console.log('✅ Image loaded successfully from URL');
    };
    
    testImg.onerror = function() {
        showError('Failed to load image from URL. Please check the link.');
        clearImagePreview();
    };
    
    testImg.src = url;
}

function previewImageFromFile() {
    const file = document.getElementById('imageFile').files[0];
    
    if (!file) {
        clearImagePreview();
        return;
    }
    
    // Validate file
    const validation = validateImageFile(file);
    if (!validation.valid) {
        showError(validation.error);
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = document.getElementById('imagePreview');
        const details = document.getElementById('imageAnalysisDetails');
        
        img.src = e.target.result;
        img.style.display = 'block';
        currentImageData = { type: 'base64', data: e.target.result };
        
        // Show analysis details
        document.getElementById('imageSource').textContent = `File: ${file.name}`;
        document.getElementById('imageStatus').textContent = `Size: ${formatFileSize(file.size)} | Type: ${file.type}`;
        details.style.display = 'block';
        
        console.log('✅ Image loaded successfully from file');
    };
    
    reader.onerror = function() {
        showError('Failed to read the selected file');
        clearImagePreview();
    };
    
    reader.readAsDataURL(file);
}

// Main Analysis Function
async function analyzeRooftop() {
    if (analysisInProgress) {
        return;
    }
    
    const button = document.querySelector('.analyze-btn');
    const resultsDiv = document.getElementById('results');
    
    // Validate inputs
    const validation = validateInputs();
    if (!validation.valid) {
        showError(validation.error);
        return;
    }
    
    // Prepare analysis data
    const analysisData = {
        imageData: currentImageData,
        address: document.getElementById('address').value || 'Unknown Location',
        electricityRate: parseFloat(document.getElementById('electricityRate').value),
        panelType: document.getElementById('panelType').value,
        roofArea: document.getElementById('roofArea').value || null
    };
    
    // Start analysis
    analysisInProgress = true;
    button.disabled = true;
    button.textContent = 'Analyzing...';
    showLoading();
    
    try {
        console.log('🔍 Starting solar analysis...', analysisData);
        
        const response = await fetch('/api/analyze-solar', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(analysisData)
        });
        
        const responseText = await response.text();
        console.log('📡 Response received:', response.status);
        
        if (!response.ok) {
            let errorMessage = 'Analysis failed';
            try {
                const errorData = JSON.parse(responseText);
                errorMessage = errorData.error || errorMessage;
            } catch (e) {
                errorMessage = `Server error (${response.status})`;
            }
            throw new Error(errorMessage);
        }
        
        const results = JSON.parse(responseText);
        console.log('✅ Analysis completed successfully');
        displayResults(results);
        
        // Save successful analysis to history
        saveAnalysisToHistory(analysisData, results);
        
    } catch (error) {
        console.error('❌ Analysis error:', error);
        showError(`Analysis failed: ${error.message}`);
    } finally {
        analysisInProgress = false;
        button.disabled = false;
        button.textContent = '🔍 Analyze Solar Potential';
    }
}

// Validation Functions
function validateInputs() {
    if (!currentImageData) {
        return { valid: false, error: 'Please provide a satellite image (URL or upload)' };
    }
    
    const electricityRate = parseFloat(document.getElementById('electricityRate').value);
    if (isNaN(electricityRate) || electricityRate <= 0 || electricityRate > 1) {
        return { valid: false, error: 'Please enter a valid electricity rate between $0.01 and $1.00 per kWh' };
    }
    
    const roofArea = document.getElementById('roofArea').value;
    if (roofArea && (parseFloat(roofArea) <= 0 || parseFloat(roofArea) > 50000)) {
        return { valid: false, error: 'Roof area must be between 1 and 50,000 sq ft' };
    }
    
    return { valid: true };
}

function validateImageFile(file) {
    const maxSize = 10 * 1024 * 1024; // 10MB
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    
    if (file.size > maxSize) {
        return { valid: false, error: 'File size must be less than 10MB' };
    }
    
    if (!allowedTypes.includes(file.type)) {
        return { valid: false, error: 'Please upload a valid image file (JPG, PNG, WebP)' };
    }
    
    return { valid: true };
}

function isValidImageUrl(url) {
    try {
        new URL(url);
        return /\.(jpg|jpeg|png|webp|gif)(\?.*)?$/i.test(url);
    } catch {
        return false;
    }
}

// Display Functions
function showLoading() {
    document.getElementById('results').innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <p>🤖 AI analyzing satellite imagery...</p>
            <p>Detecting roof segments, obstacles, and optimal panel placement...</p>
        </div>
    `;
}

function displayResults(data) {
    const resultsHTML = `
        <div class="results-grid">
            <div class="result-card">
                <h3>🏠 Roof Analysis</h3>
                <div class="metric">
                    <span class="metric-label">Detected Area:</span>
                    <span class="metric-value">${data.roofAnalysis.detectedArea.toLocaleString()} sq ft</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Usable Area:</span>
                    <span class="metric-value">${data.roofAnalysis.usableArea.toLocaleString()} sq ft</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Orientation:</span>
                    <span class="metric-value">${capitalizeFirst(data.roofAnalysis.orientation)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Shading Factor:</span>
                    <span class="metric-value">${data.roofAnalysis.shading}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Roof Condition:</span>
                    <span class="metric-value">${capitalizeFirst(data.roofAnalysis.condition)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Obstacles:</span>
                    <span class="metric-value">${data.roofAnalysis.obstacles.map(capitalizeFirst).join(', ')}</span>
                </div>
            </div>
            
            <div class="result-card">
                <h3>⚡ System Design</h3>
                <div class="metric">
                    <span class="metric-label">System Capacity:</span>
                    <span class="metric-value">${data.systemSpecs.capacity} kW</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Panel Count:</span>
                    <span class="metric-value">${data.systemSpecs.panelCount}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Panel Type:</span>
                    <span class="metric-value">${data.systemSpecs.panelType}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Panel Efficiency:</span>
                    <span class="metric-value">${(data.systemSpecs.efficiency * 100).toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Annual Production:</span>
                    <span class="metric-value">${data.energyProduction.annualProduction.toLocaleString()} kWh</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Capacity Factor:</span>
                    <span class="metric-value">${data.energyProduction.capacityFactor}%</span>
                </div>
            </div>
            
            <div class="result-card">
                <h3>💰 Financial Analysis</h3>
                <div class="metric">
                    <span class="metric-label">Gross System Cost:</span>
                    <span class="metric-value">$${data.financial.grossCost.toLocaleString()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">After Tax Credit:</span>
                    <span class="metric-value">$${data.financial.netCost.toLocaleString()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Annual Savings:</span>
                    <span class="metric-value">$${data.financial.annualSavings.toLocaleString()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Monthly Savings:</span>
                    <span class="metric-value">$${Math.round(data.financial.annualSavings / 12).toLocaleString()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Payback Period:</span>
                    <span class="metric-value">${data.financial.paybackPeriod} years</span>
                </div>
                <div class="metric">
                    <span class="metric-label">25-Year ROI:</span>
                    <span class="metric-value">${data.financial.roiPercentage}%</span>
                </div>
            </div>
            
            <div class="result-card">
                <h3>🌱 Environmental Impact</h3>
                <div class="metric">
                    <span class="metric-label">Annual CO₂ Offset:</span>
                    <span class="metric-value">${data.environmental.annualCo2Offset} tons</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Trees Equivalent:</span>
                    <span class="metric-value">${data.environmental.treesEquivalent}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Cars Off Road:</span>
                    <span class="metric-value">${data.environmental.carsEquivalent}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">25-Year CO₂ Savings:</span>
                    <span class="metric-value">${data.environmental.lifetimeCo2Offset} tons</span>
                </div>
            </div>
        </div>
        
        <div class="confidence-score">
            Analysis Confidence: ${data.confidence}% | Method: ${data.analysisMethod}
            <br><small>Analysis completed on ${new Date(data.timestamp).toLocaleString()}</small>
        </div>
    `;
    
    document.getElementById('results').innerHTML = resultsHTML;
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'start' 
    });
}

function showError(message) {
    document.getElementById('results').innerHTML = `
        <div class="error-message">
            ⚠️ ${message}
            <br><br>
            <strong>Troubleshooting tips:</strong>
            <ul>
                <li>Ensure the image shows a clear rooftop view</li>
                <li>Try using a different image URL or upload the file directly</li>
                <li>Check that all required fields are filled correctly</li>
                <li>Verify your internet connection</li>
            </ul>
        </div>
    `;
}

// Utility Functions
function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getUrlDomain(url) {
    try {
        return new URL(url).hostname;
    } catch {
        return 'External URL';
    }
}

function saveAnalysisToHistory(inputData, results) {
    try {
        const history = JSON.parse(localStorage.getItem('solarAnalyzer_history') || '[]');
        const analysisRecord = {
            timestamp: new Date().toISOString(),
            address: inputData.address,
            results: {
                capacity: results.systemSpecs.capacity,
                annualProduction: results.energyProduction.annualProduction,
                paybackPeriod: results.financial.paybackPeriod,
                confidence: results.confidence
            }
        };
        
        history.unshift(analysisRecord);
        if (history.length > 10) history.pop(); // Keep only last 10
        
        localStorage.setItem('solarAnalyzer_history', JSON.stringify(history));
    } catch (error) {
        console.warn('Failed to save analysis to history:', error);
    }
}

// Export functions for potential external use
window.SolarAnalyzer = {
    analyzeRooftop,
    switchTab,
    previewImageFromUrl,
    previewImageFromFile
};
