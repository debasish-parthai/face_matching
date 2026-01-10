// Global variables
let candidateFile = null;
let referenceFiles = [];

// DOM elements
const candidateFileInput = document.getElementById('candidate-file');
const referenceFilesInput = document.getElementById('reference-files');
const candidatePreview = document.getElementById('candidate-preview');
const candidateImg = document.getElementById('candidate-img');
const referencePreviews = document.getElementById('reference-previews');
const compareBtn = document.getElementById('compare-btn');
const progressBar = document.getElementById('progress-bar');
const resultsSection = document.getElementById('results-section');
const jsonOutput = document.getElementById('json-output');

// Initialize event listeners
document.addEventListener('DOMContentLoaded', function() {
    setupFileInputs();
    updateCompareButton();
});

// Setup file input event listeners
function setupFileInputs() {
    // Candidate file input
    candidateFileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            candidateFile = file;
            displayCandidatePreview(file);
        }
        updateCompareButton();
    });

    // Reference files input
    referenceFilesInput.addEventListener('change', function(e) {
        const files = Array.from(e.target.files);
        referenceFiles = files;
        displayReferencePreviews(files);
        updateCompareButton();
    });
}

// Display candidate image preview
function displayCandidatePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        candidateImg.src = e.target.result;
        candidatePreview.style.display = 'block';
        document.querySelector('#candidate-upload .upload-content').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Display reference images preview
function displayReferencePreviews(files) {
    referencePreviews.innerHTML = '';

    files.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            const previewDiv = document.createElement('div');
            previewDiv.className = 'reference-preview';

            previewDiv.innerHTML = `
                <img src="${e.target.result}" alt="Reference ${index + 1}">
                <button class="remove-btn" onclick="removeReferenceFile(${index})">×</button>
            `;

            referencePreviews.appendChild(previewDiv);
        };
        reader.readAsDataURL(file);
    });

    // Hide upload content if files are selected
    if (files.length > 0) {
        document.querySelector('#reference-upload .upload-content').style.display = 'none';
    } else {
        document.querySelector('#reference-upload .upload-content').style.display = 'block';
    }
}

// Remove candidate file
function removeFile(type) {
    if (type === 'candidate') {
        candidateFile = null;
        candidateFileInput.value = '';
        candidatePreview.style.display = 'none';
        document.querySelector('#candidate-upload .upload-content').style.display = 'block';
    }
    updateCompareButton();
}

// Remove reference file
function removeReferenceFile(index) {
    referenceFiles.splice(index, 1);
    referenceFilesInput.value = ''; // Reset input to allow re-selection
    displayReferencePreviews(referenceFiles);
    updateCompareButton();
}

// Update compare button state
function updateCompareButton() {
    const isEnabled = candidateFile && referenceFiles.length > 0;
    compareBtn.disabled = !isEnabled;

    if (isEnabled) {
        compareBtn.innerHTML = '<span class="btn-text">Compare Faces</span><div class="loading-spinner" style="display: none;"></div>';
    }
}

// Perform face matching
async function performFaceMatching() {
    if (!candidateFile || referenceFiles.length === 0) {
        alert('Please select both candidate and reference images.');
        return;
    }

    // Show loading state
    setLoadingState(true);
    updateProgress(0);

    try {
        // Create FormData
        const formData = new FormData();
        formData.append('candidate_file', candidateFile);

        referenceFiles.forEach(file => {
            formData.append('reference_files', file);
        });

        formData.append('save_cropped_faces', 'true');

        updateProgress(20);

        // Make API request
        const response = await fetch('http://localhost:8000/match-faces/', {
            method: 'POST',
            body: formData
        });

        updateProgress(60);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const results = await response.json();

        updateProgress(80);

        // Display results
        displayResults(results);

        updateProgress(100);

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during face matching. Please check that the backend server is running.');
    } finally {
        setLoadingState(false);
    }
}

// Set loading state
function setLoadingState(isLoading) {
    const btnText = compareBtn.querySelector('.btn-text');
    const spinner = compareBtn.querySelector('.loading-spinner');

    if (isLoading) {
        btnText.textContent = 'Processing...';
        spinner.style.display = 'block';
        compareBtn.disabled = true;
        progressBar.style.display = 'block';
    } else {
        btnText.textContent = 'Compare Faces';
        spinner.style.display = 'none';
        compareBtn.disabled = false;
        progressBar.style.display = 'none';
    }
}

// Update progress bar
function updateProgress(percent) {
    const progressFill = progressBar.querySelector('.progress-fill');
    progressFill.style.width = `${percent}%`;
}

// Display results
function displayResults(results) {
    resultsSection.style.display = 'block';

    // Display candidate face
    displayCandidateFace(results);

    // Display reference faces
    displayReferenceFaces(results);

    // Display JSON results
    jsonOutput.textContent = JSON.stringify(results, null, 2);

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Display candidate face
function displayCandidateFace(results) {
    const candidateFaceDisplay = document.getElementById('candidate-face-display');

    if (results.candidate_cropped_image_base64) {
        candidateFaceDisplay.innerHTML = `
            <div class="face-item">
                <img src="${results.candidate_cropped_image_base64}" alt="Candidate face">
                <div class="face-info">
                    <strong>Candidate</strong><br>
                    ${results.candidate_faces_detected} face(s) detected
                </div>
            </div>
        `;
    } else {
        candidateFaceDisplay.innerHTML = '<p class="no-face">No face detected in candidate image</p>';
    }
}

// Display reference faces
function displayReferenceFaces(results) {
    const referenceFacesDisplay = document.getElementById('reference-faces-display');
    referenceFacesDisplay.innerHTML = '';

    let hasFaces = false;

    results.reference_results.forEach((ref, index) => {
        if (ref.face_attributes && ref.face_attributes.length > 0) {
            ref.face_attributes.forEach((face, faceIndex) => {
                if (face.cropped_image_base64) {
                    hasFaces = true;
                    const faceDiv = document.createElement('div');
                    faceDiv.className = 'face-item';

                    const age = face.age !== undefined ? `Age: ~${Math.round(face.age)}` : '';
                    const gender = face.gender !== undefined ? `Gender: ${face.gender === 0 ? 'Female' : 'Male'}` : '';
                    const score = `Score: ${face.score.toFixed(2)}/100`;

                    faceDiv.innerHTML = `
                        <img src="${face.cropped_image_base64}" alt="Reference ${index + 1} face ${faceIndex + 1}">
                        <div class="face-info">
                            <strong>Ref ${ref.reference_index}</strong><br>
                            ${age}<br>
                            ${gender}<br>
                            ${score}
                        </div>
                    `;

                    referenceFacesDisplay.appendChild(faceDiv);
                }
            });
        }
    });

    if (!hasFaces) {
        referenceFacesDisplay.innerHTML = '<p class="no-face">No faces detected in reference images</p>';
    }
}

// Copy results to clipboard
function copyResults() {
    const jsonText = jsonOutput.textContent;
    navigator.clipboard.writeText(jsonText).then(function() {
        // Show temporary success message
        const copyBtn = document.querySelector('.copy-btn');
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        copyBtn.style.background = '#2ed573';

        setTimeout(() => {
            copyBtn.textContent = originalText;
            copyBtn.style.background = '';
        }, 2000);
    }).catch(function(err) {
        console.error('Could not copy text: ', err);
        alert('Failed to copy to clipboard');
    });
}

// Event listener for compare button
compareBtn.addEventListener('click', performFaceMatching);
