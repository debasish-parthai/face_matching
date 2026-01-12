from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
import shutil
import tempfile
from face_matcher import FaceMatcherInsightFace
from face_matcher_optimize_for_script import ImprovedFaceMatcherInsightFace
import logging
import base64

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string for frontend display"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        return ""

app = FastAPI(
    title="Face Matching API",
    description="API for face matching using InsightFace",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize face matcher
face_matcher = ImprovedFaceMatcherInsightFace()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Face Matching API",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy"
    }

@app.post("/match-faces/")
async def match_faces(
    candidate_file: UploadFile = File(...),
    reference_files: List[UploadFile] = File(...),
    save_cropped_faces: bool = Form(True)
):
    """
    Match a candidate face with multiple reference faces.

    - **candidate_file**: Image file containing the candidate face
    - **reference_files**: List of reference image files (up to 4 recommended)
    """
    if len(reference_files) == 0:
        raise HTTPException(status_code=400, detail="At least one reference file is required")

    if len(reference_files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 reference files allowed")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create subdirectory for cropped faces
            cropped_dir = os.path.join(temp_dir, "cropped_faces")
            os.makedirs(cropped_dir, exist_ok=True)

            # Save candidate file
            candidate_path = os.path.join(temp_dir, "candidate.jpg")
            with open(candidate_path, "wb") as f:
                shutil.copyfileobj(candidate_file.file, f)

            # Save reference files
            reference_paths = []
            for i, ref_file in enumerate(reference_files):
                ref_path = os.path.join(temp_dir, f"reference_{i}.jpg")
                with open(ref_path, "wb") as f:
                    shutil.copyfileobj(ref_file.file, f)
                reference_paths.append(ref_path)

            # Perform matching
            results = face_matcher.match_candidate_with_references(
                candidate_path, reference_paths,
                save_cropped_faces=save_cropped_faces,
                output_dir=cropped_dir
            )

            # Convert cropped image paths to base64 for frontend display
            if 'candidate_cropped_image_path' in results and results['candidate_cropped_image_path']:
                results['candidate_cropped_image_base64'] = image_to_base64(results['candidate_cropped_image_path'])

            for ref_result in results['reference_results']:
                for face_attr in ref_result['face_attributes']:
                    if 'cropped_image_path' in face_attr and face_attr['cropped_image_path']:
                        face_attr['cropped_image_base64'] = image_to_base64(face_attr['cropped_image_path'])

            # Convert all numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                else:
                    return obj

            results = convert_numpy_types(results)

            return JSONResponse(content=results)

        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/extract-faces/")
async def extract_faces(file: UploadFile = File(...)):
    """
    Extract faces from an image.

    - **file**: Image file to extract faces from
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Save file
            file_path = os.path.join(temp_dir, "image.jpg")
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # Extract faces
            faces = face_matcher.extract_faces(file_path)

            # Convert numpy arrays to lists for JSON serialization
            for face in faces:
                face['embedding'] = face['embedding'].tolist() if hasattr(face['embedding'], 'tolist') else face['embedding']
                face['bbox'] = face['bbox'].tolist() if hasattr(face['bbox'], 'tolist') else face['bbox']

            return {
                "faces_detected": len(faces),
                "faces": faces
            }

        except Exception as e:
            logger.error(f"Error extracting faces: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting faces: {str(e)}")

@app.post("/compare-embeddings/")
async def compare_embeddings(
    embedding1: List[float],
    embedding2: List[float]
):
    """
    Compare two face embeddings.

    - **embedding1**: First face embedding
    - **embedding2**: Second face embedding
    """
    try:
        import numpy as np

        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        similarity = face_matcher.calculate_similarity(emb1, emb2)

        return {
            "similarity_score": similarity,
            "is_same_person": similarity > 30.0  # InsightFace threshold
        }

    except Exception as e:
        logger.error(f"Error comparing embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing embeddings: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
