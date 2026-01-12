from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
import os
import shutil
import tempfile
import numpy as np
from face_matcher import FaceMatcherInsightFace
from face_matcher_optimize_for_script import ImprovedFaceMatcherInsightFace
import logging
import base64
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

USERS_FINAL_DIR = os.path.join(os.path.dirname(__file__), "TestQA", "Users_Final")

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

@app.get("/cross-user-matching-results")
async def get_cross_user_matching_results():
    """Get the pre-calculated results of cross-user matching analysis"""
    results_path = os.path.join(os.path.dirname(__file__), "TestQA", "cross_user_matching_results.json")
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Analysis results not found. Run cross_user_matching_analysis.py first.")
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading results: {str(e)}")

@app.post("/match-against-database/")
async def match_against_database(
    file: UploadFile = File(...),
    top_n: int = 10,
    threshold: float = 30.0
):
    """
    Match an uploaded face against all users in the stored database (Users_Final).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Save uploaded file
            file_path = os.path.join(temp_dir, "candidate.jpg")
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # Extract faces from candidate
            candidate_faces = face_matcher.extract_faces(file_path)
            if not candidate_faces:
                return {"message": "No face detected in uploaded image", "results": []}
            
            # Use the best face detected
            candidate_face = face_matcher.get_best_candidate_face(candidate_faces)
            candidate_emb = candidate_face['embedding']
            
            match_results = []
            
            # Load and compare against all users in Users_Final
            if not os.path.exists(USERS_FINAL_DIR):
                raise HTTPException(status_code=500, detail="Users database directory not found")
                
            for user_folder in os.listdir(USERS_FINAL_DIR):
                user_path = os.path.join(USERS_FINAL_DIR, user_folder)
                metadata_path = os.path.join(user_path, "metadata.json")
                
                if not os.path.isdir(user_path) or not os.path.exists(metadata_path):
                    continue
                    
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                user_best_score = 0.0
                user_best_photo = ""
                
                # Check primary photo
                if metadata.get("primary_photo") and metadata["primary_photo"].get("embedding_path"):
                    emb_path = os.path.join(user_path, metadata["primary_photo"]["embedding_path"])
                    if os.path.exists(emb_path):
                        ref_emb = np.load(emb_path)
                        score = face_matcher.calculate_similarity(candidate_emb, ref_emb)
                        if score > user_best_score:
                            user_best_score = score
                            user_best_photo = metadata["primary_photo"]["cropped_path"]
                
                # Check additional photos
                for photo in metadata.get("additional_photos", []):
                    if photo.get("embedding_path"):
                        emb_path = os.path.join(user_path, photo["embedding_path"])
                        if os.path.exists(emb_path):
                            ref_emb = np.load(emb_path)
                            score = face_matcher.calculate_similarity(candidate_emb, ref_emb)
                            if score > user_best_score:
                                user_best_score = score
                                user_best_photo = photo["cropped_path"]
                
                if user_best_score >= threshold:
                    match_results.append({
                        "user_id": metadata["user_id"],
                        "folder": user_folder,
                        "best_score": round(user_best_score, 2),
                        "best_photo": user_best_photo
                    })
            
            # Sort and limit
            match_results.sort(key=lambda x: x["best_score"], reverse=True)
            match_results = match_results[:top_n]
            
            return {
                "candidate_faces_detected": len(candidate_faces),
                "top_matches": match_results
            }
            
        except Exception as e:
            logger.error(f"Error in database matching: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
