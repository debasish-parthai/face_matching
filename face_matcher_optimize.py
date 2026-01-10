import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import os

class ImprovedFaceMatcherInsightFace:
    def __init__(self, model_name: str = 'buffalo_l', use_gpu: bool = False):
        """
        model_name: 'buffalo_l' (best), 'buffalo_m', 'buffalo_s' (fastest)
        use_gpu: Set True if CUDA is available for faster processing
        """
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

        self.app = FaceAnalysis(
            name=model_name,
            providers=providers
        )
        # Optimized detection size - smaller is faster
        self.app.prepare(ctx_id=0, det_size=(320, 320))  # Changed from 640x640

        # Cache for loaded images to avoid reprocessing
        self.image_cache = {}

    def preprocess_image(self, image_path: str, max_size: int = 1024) -> Optional[np.ndarray]:
        """Resize large images for faster processing"""
        # Check cache first
        if image_path in self.image_cache:
            return self.image_cache[image_path]

        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Could not read image: {image_path}")
            return None

        # Resize if too large
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logging.debug(f"Resized {image_path} from {w}x{h} to {new_w}x{new_h}")

        # Cache the processed image
        self.image_cache[image_path] = image
        return image

    def calculate_blur_laplacian(self, image: np.ndarray, bbox: List) -> float:
        """Calculate blur score using Laplacian (optimized)"""
        x1, y1, x2, y2 = map(int, bbox[:4])

        # Add bounds checking
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            return 0.0

        # Convert to grayscale only once
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # Use smaller kernel for faster processing
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        return laplacian.var()

    def is_face_viable(self, image: np.ndarray, face_info: Dict,
                       blur_threshold: float = 30.0,  # Lowered for better recall
                       min_face_size: int = 30,  # Reduced minimum size
                       min_det_score: float = 0.25) -> bool:  # Lowered threshold
        """Check face quality and detection confidence (optimized thresholds)"""
        bbox = face_info['bbox']

        # Check detection score first (fastest check)
        det_score = face_info['det_score']
        if det_score < min_det_score:
            return False

        # Check face size
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width < min_face_size or height < min_face_size:
            return False

        # Blur check last (most expensive)
        blur_score = self.calculate_blur_laplacian(image, bbox)
        if blur_score < blur_threshold:
            return False

        return True

    def extract_faces(self, image_path: str,
                     use_preprocessing: bool = True) -> List[Dict]:
        """Extract all viable faces with embeddings (optimized)"""
        try:
            # Use preprocessed image
            if use_preprocessing:
                image = self.preprocess_image(image_path)
            else:
                image = cv2.imread(image_path)

            if image is None:
                return []

            # Detect and extract faces
            faces = self.app.get(image)

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

            return viable_faces

        except Exception as e:
            logging.error(f"Error extracting faces from {image_path}: {str(e)}")
            return []

    def calculate_similarity(self, embedding1: np.ndarray,
                           embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between embeddings (optimized)
        Returns score 0-100
        """
        # Embeddings from InsightFace are already normalized
        # But we normalize again to be safe
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity using optimized dot product
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Convert to 0-100 scale
        # InsightFace threshold: >0.3-0.4 = same person
        score = max(0, min(100, similarity * 100))

        return score

    def get_best_candidate_face(self, faces: List[Dict]) -> Optional[Dict]:
        """Select the best face from multiple detections"""
        if not faces:
            return None

        # Prioritize: highest detection score + largest face
        best_face = max(faces, key=lambda f: (
            f['det_score'] * 0.7 +  # Detection confidence weight
            ((f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1])) / 10000 * 0.3  # Size weight
        ))

        return best_face

    def match_candidate_with_references(self,
                                       candidate_path: str,
                                       reference_paths: List[str],
                                       use_parallel: bool = True,
                                       match_threshold: float = 40.0) -> Dict[str, any]:
        """
        Match candidate photo with reference photos (optimized)

        Args:
            use_parallel: Process reference images in parallel
            match_threshold: Minimum score to consider a match
        """
        results = {
            'candidate_faces_detected': 0,
            'best_candidate_used': False,
            'reference_results': [],
            'is_match': False,
            'avg_match_score': 0.0
        }

        # Extract candidate face
        candidate_faces = self.extract_faces(candidate_path)

        if not candidate_faces:
            logging.warning("No viable face in candidate photo")
            return results

        results['candidate_faces_detected'] = len(candidate_faces)

        # Use best quality face if multiple detected
        if len(candidate_faces) > 1:
            candidate_face = self.get_best_candidate_face(candidate_faces)
            results['best_candidate_used'] = True
        else:
            candidate_face = candidate_faces[0]

        candidate_embedding = candidate_face['embedding']

        # Process references
        if use_parallel and len(reference_paths) > 1:
            # Parallel processing for multiple references
            with ThreadPoolExecutor(max_workers=min(4, len(reference_paths))) as executor:
                ref_results = list(executor.map(
                    lambda ref_path: self._process_reference(
                        ref_path, candidate_embedding, reference_paths.index(ref_path)
                    ),
                    reference_paths
                ))
            results['reference_results'] = ref_results
        else:
            # Sequential processing
            for idx, ref_path in enumerate(reference_paths):
                ref_result = self._process_reference(ref_path, candidate_embedding, idx)
                results['reference_results'].append(ref_result)

        # Calculate match statistics
        valid_scores = [r['max_score'] for r in results['reference_results'] if r['max_score'] > 0]

        if valid_scores:
            results['avg_match_score'] = sum(valid_scores) / len(valid_scores)
            results['is_match'] = results['avg_match_score'] >= match_threshold
            results['matches_above_threshold'] = sum(1 for s in valid_scores if s >= match_threshold)

        return results

    def _process_reference(self, ref_path: str,
                          candidate_embedding: np.ndarray,
                          idx: int) -> Dict:
        """Process a single reference image"""
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
            return ref_result

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

        return ref_result

    def clear_cache(self):
        """Clear image cache to free memory"""
        self.image_cache.clear()


# Usage Example
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize with optimized settings
    matcher = ImprovedFaceMatcherInsightFace(
        model_name='buffalo_l',  # Use 'buffalo_s' for faster processing
        use_gpu=False  # Set True if you have CUDA
    )

    # Match with optimizations enabled
    results = matcher.match_candidate_with_references(
        "candidate_selfie.jpg",
        ["biodata1.jpg", "biodata2.jpg", "biodata3.jpg", "biodata4.jpg"],
        use_parallel=True,  # Process references in parallel
        match_threshold=40.0  # Adjust based on your needs
    )

    # Display results
    print(f"Candidate faces detected: {results['candidate_faces_detected']}")
    print(f"Overall match: {'YES' if results['is_match'] else 'NO'}")
    print(f"Average match score: {results.get('avg_match_score', 0):.2f}/100")
    print(f"References above threshold: {results.get('matches_above_threshold', 0)}/4\n")

    for ref in results['reference_results']:
        print(f"Reference {ref['reference_index']} ({os.path.basename(ref['reference_path'])}):")
        print(f"  Faces detected: {ref['faces_detected']}")
        print(f"  Best match score: {ref['max_score']:.2f}/100")
        if ref['face_attributes']:
            for i, attr in enumerate(ref['face_attributes']):
                print(f"    Face {i+1}: Age~{attr['age']}, Gender: {attr['gender']}, Score: {attr['score']:.2f}")
        print()

    # Clear cache when done
    matcher.clear_cache()
