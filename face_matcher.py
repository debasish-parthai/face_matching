import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from typing import List, Dict
import logging

class FaceMatcherInsightFace:
    def __init__(self, model_name: str = 'buffalo_l'):
        """
        model_name: 'buffalo_l' (best), 'buffalo_m', 'buffalo_s' (fastest)
        """
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CPUExecutionProvider']  # Use 'CUDAExecutionProvider' for GPU
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def calculate_blur_laplacian(self, image: np.ndarray, bbox: List) -> float:
        """Calculate blur score using Laplacian"""
        x1, y1, x2, y2 = map(int, bbox[:4])
        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def is_face_viable(self, image: np.ndarray, face_info: Dict,
                       blur_threshold: float = 50.0,
                       min_face_size: int = 40,
                       min_det_score: float = 0.3) -> bool:
        """Check face quality and detection confidence"""
        bbox = face_info['bbox']

        # Check detection score
        det_score = face_info['det_score']
        if det_score < min_det_score:
            logging.debug(f"Face rejected: detection score {det_score:.3f} < {min_det_score}")
            return False

        # Check face size
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width < min_face_size or height < min_face_size:
            logging.debug(f"Face rejected: size {width}x{height} < {min_face_size}px")
            return False

        # Check blur
        blur_score = self.calculate_blur_laplacian(image, bbox)
        if blur_score < blur_threshold:
            logging.debug(f"Face rejected: blur score {blur_score:.1f} < {blur_threshold}")
            return False

        return True

    def extract_faces(self, image_path: str) -> List[Dict]:
        """Extract all viable faces with embeddings"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Could not read image: {image_path}")
                return []

            # Detect and extract faces
            faces = self.app.get(image)
            logging.debug(f"Image {image_path}: {len(faces)} faces detected by InsightFace")

            viable_faces = []
            for face in faces:
                if self.is_face_viable(image, face):
                    viable_faces.append({
                        'embedding': face['embedding'],
                        'bbox': face['bbox'],
                        'det_score': face['det_score'],
                        'age': face.get('age'),
                        'gender': face.get('gender')
                    })

            logging.debug(f"Image {image_path}: {len(viable_faces)} viable faces after filtering")
            return viable_faces
        except Exception as e:
            logging.error(f"Error extracting faces from {image_path}: {str(e)}")
            return []

    def calculate_similarity(self, embedding1: np.ndarray,
                           embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between embeddings
        Returns score 0-100
        """
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)

        # Convert to 0-100 scale
        # InsightFace typically: >0.3 = same person, >0.5 = high confidence
        score = max(0, min(100, similarity * 100))

        return score

    def match_candidate_with_references(self,
                                       candidate_path: str,
                                       reference_paths: List[str]) -> Dict[str, any]:
        """Match candidate photo with 4 reference photos"""
        results = {
            'candidate_faces_detected': 0,
            'reference_results': []
        }

        # Extract candidate face
        candidate_faces = self.extract_faces(candidate_path)

        if not candidate_faces:
            logging.warning("No viable face in candidate photo")
            return results

        results['candidate_faces_detected'] = len(candidate_faces)
        candidate_embedding = candidate_faces[0]['embedding']

        # Match with each reference photo
        for idx, ref_path in enumerate(reference_paths):
            ref_result = {
                'reference_index': idx + 1,
                'reference_path': ref_path,
                'faces_detected': 0,
                'max_score': 0.0,
                'all_scores': [],
                'face_attributes': []
            }

            ref_faces = self.extract_faces(ref_path)
            ref_result['faces_detected'] = len(ref_faces)

            if not ref_faces:
                results['reference_results'].append(ref_result)
                continue

            # Match with all faces in reference
            for ref_face in ref_faces:
                score = self.calculate_similarity(
                    candidate_embedding,
                    ref_face['embedding']
                )
                ref_result['all_scores'].append(score)
                ref_result['face_attributes'].append({
                    'age': ref_face.get('age'),
                    'gender': ref_face.get('gender'),
                    'score': score
                })

            if ref_result['all_scores']:
                ref_result['max_score'] = max(ref_result['all_scores'])

            results['reference_results'].append(ref_result)

        return results


# Usage
if __name__ == "__main__":
    matcher = FaceMatcherInsightFace(model_name='buffalo_l')

    results = matcher.match_candidate_with_references(
        "candidate_selfie.jpg",
        ["biodata1.jpg", "biodata2.jpg", "biodata3.jpg", "biodata4.jpg"]
    )

    print(f"Candidate faces: {results['candidate_faces_detected']}")
    for ref in results['reference_results']:
        print(f"\nReference {ref['reference_index']}:")
        print(f"  Faces detected: {ref['faces_detected']}")
        print(f"  Best match score: {ref['max_score']:.2f}/100")
        if ref['face_attributes']:
            for i, attr in enumerate(ref['face_attributes']):
                print(f"  Face {i+1}: Age~{attr['age']}, Score: {attr['score']:.2f}")
