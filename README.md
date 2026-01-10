# Face Matching Project

A face matching application using InsightFace and FastAPI for comparing faces in images.

## Features

- Face detection and feature extraction using InsightFace
- Face quality assessment (blur, size, confidence)
- Cosine similarity matching between face embeddings
- REST API with FastAPI
- Automatic documentation with Swagger UI
- Support for multiple reference images

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install InsightFace

**Windows Users:** You may need to install Microsoft Visual C++ Build Tools first.

```bash
pip install insightface
```

If you encounter build issues, try:

```bash
# Using conda (if available)
conda install -c conda-forge insightface

# Or use pre-built wheels from alternative sources
pip install insightface --index-url https://pypi.org/simple/
```

## Project Structure

```
face-matching-project/
├── venv/                     # Virtual environment
├── face_matcher.py          # Main face matching logic
├── app.py                   # FastAPI application
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── test_images/            # (Optional) Test images directory
```

## Usage

### Running the API Server

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Root**: http://localhost:8000/

### API Endpoints

#### 1. Health Check
- **GET** `/health`
- Check if the service and InsightFace are available

#### 2. Match Faces
- **POST** `/match-faces/`
- Match a candidate face with multiple reference faces
- **Parameters:**
  - `candidate_file`: Image file with candidate face
  - `reference_files`: List of reference image files

#### 3. Extract Faces
- **POST** `/extract-faces/`
- Extract faces from a single image
- **Parameters:**
  - `file`: Image file to analyze

#### 4. Compare Embeddings
- **POST** `/compare-embeddings/`
- Compare two face embeddings directly
- **Parameters:**
  - `embedding1`: First face embedding (list of floats)
  - `embedding2`: Second face embedding (list of floats)

## Testing with FastAPI Docs

1. Start the server: `python app.py`
2. Open http://localhost:8000/docs in your browser
3. Use the interactive Swagger UI to test endpoints:
   - Upload images for face matching
   - View extracted face data
   - Test embedding comparisons

## Usage Examples

### Python Script Usage

```python
from face_matcher import FaceMatcherInsightFace

# Initialize matcher
matcher = FaceMatcherInsightFace(model_name='buffalo_l')

# Match candidate with references
results = matcher.match_candidate_with_references(
    "candidate_selfie.jpg",
    ["biodata1.jpg", "biodata2.jpg", "biodata3.jpg", "biodata4.jpg"]
)

print(f"Candidate faces: {results['candidate_faces_detected']}")
for ref in results['reference_results']:
    print(f"Reference {ref['reference_index']}: {ref['max_score']:.2f}/100 match")
```

### API Usage with curl

```bash
# Match faces
curl -X POST "http://localhost:8000/match-faces/" \
  -F "candidate_file=@candidate.jpg" \
  -F "reference_files=@ref1.jpg" \
  -F "reference_files=@ref2.jpg"

# Extract faces
curl -X POST "http://localhost:8000/extract-faces/" \
  -F "file=@image.jpg"
```

### API Usage with Python requests

```python
import requests

# Match faces
files = {
    'candidate_file': open('candidate.jpg', 'rb'),
    'reference_files': [
        open('ref1.jpg', 'rb'),
        open('ref2.jpg', 'rb')
    ]
}

response = requests.post('http://localhost:8000/match-faces/', files=files)
print(response.json())
```

## Face Matching Logic

### Face Quality Checks

- **Detection Confidence**: Minimum score of 0.5
- **Face Size**: Minimum 80x80 pixels
- **Blur Detection**: Laplacian variance > 100

### Similarity Scoring

- Uses cosine similarity between face embeddings
- Scores range from 0-100
- Typically >30 indicates same person, >50 indicates high confidence

### Model Options

- `buffalo_l`: Best accuracy (default)
- `buffalo_m`: Balanced performance
- `buffalo_s`: Fastest, least accurate

## Dependencies

- **insightface**: Face detection and recognition
- **opencv-python**: Image processing
- **onnxruntime**: Model inference
- **fastapi**: Web framework
- **uvicorn**: ASGI server
- **numpy**: Numerical computations

## Troubleshooting

### InsightFace Installation Issues

**Windows:**
- Install Microsoft Visual C++ Build Tools
- Or use: `pip install insightface --no-build-isolation`

**Linux/Mac:**
- Install build dependencies: `sudo apt-get install build-essential`

### GPU Support

For GPU acceleration, change provider to 'CUDAExecutionProvider':

```python
self.app = FaceAnalysis(
    name=model_name,
    providers=['CUDAExecutionProvider']
)
```

### Memory Issues

- Use smaller models (`buffalo_s`) for limited RAM
- Process images sequentially for large batches

## License

This project is open source. Check individual package licenses for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request
