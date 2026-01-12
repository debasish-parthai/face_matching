import os
import json
import numpy as np
from typing import List, Dict, Tuple
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_user_data(users_dir: str) -> List[Dict]:
    """Load all user metadata and embeddings from the users directory"""
    users_data = []
    
    if not os.path.exists(users_dir):
        logger.error(f"Directory not found: {users_dir}")
        return []

    user_folders = [f for f in os.listdir(users_dir) if os.path.isdir(os.path.join(users_dir, f))]
    logger.info(f"Found {len(user_folders)} user folders in {users_dir}")

    for folder in user_folders:
        user_path = os.path.join(users_dir, folder)
        metadata_path = os.path.join(user_path, "metadata.json")
        
        if not os.path.exists(metadata_path):
            continue
            
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Load embeddings
            embeddings = []
            
            # Primary photo embedding
            if metadata.get("primary_photo") and metadata["primary_photo"].get("embedding_path"):
                emb_path = os.path.join(user_path, metadata["primary_photo"]["embedding_path"])
                if os.path.exists(emb_path):
                    embeddings.append({
                        "file_id": metadata["primary_photo"]["cropped_path"],
                        "embedding": np.load(emb_path)
                    })
            
            # Additional photos embeddings
            for photo in metadata.get("additional_photos", []):
                if photo.get("embedding_path"):
                    emb_path = os.path.join(user_path, photo["embedding_path"])
                    if os.path.exists(emb_path):
                        embeddings.append({
                            "file_id": photo["cropped_path"],
                            "embedding": np.load(emb_path)
                        })
            
            if embeddings:
                users_data.append({
                    "user_id": metadata["user_id"],
                    "user_number": metadata["user_number"],
                    "folder": folder,
                    "embeddings": embeddings
                })
        except Exception as e:
            logger.error(f"Error loading data for {folder}: {str(e)}")
            
    return users_data

def calculate_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    # InsightFace embeddings are usually normalized, but we normalize to be sure
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = np.dot(feat1, feat2) / (norm1 * norm2)
    # Convert to 0-100 scale (InsightFace style)
    return float(max(0, min(100, similarity * 100)))

def run_cross_user_matching():
    users_dir = os.path.join(os.path.dirname(__file__), "TestQA", "Users_Final")
    logger.info(f"Starting cross-user matching analysis on {users_dir}...")
    
    users = load_user_data(users_dir)
    num_users = len(users)
    logger.info(f"Loaded {num_users} users with embeddings.")
    
    all_pairs_results = []
    
    # Compare every user with every other user
    for i in range(num_users):
        user1 = users[i]
        
        if i % 10 == 0:
            logger.info(f"Processing user {i+1}/{num_users}...")
            
        for j in range(i + 1, num_users):
            user2 = users[j]
            
            best_score = 0.0
            best_pair = None
            
            # Compare all photos of user1 with all photos of user2
            for emb1_data in user1["embeddings"]:
                for emb2_data in user2["embeddings"]:
                    score = calculate_similarity(emb1_data["embedding"], emb2_data["embedding"])
                    
                    if score > best_score:
                        best_score = score
                        best_pair = (emb1_data["file_id"], emb2_data["file_id"])
            
            # Only record if we found a match
            if best_pair:
                all_pairs_results.append({
                    "user1_id": user1["user_id"],
                    "user1_folder": user1["folder"],
                    "user2_id": user2["user_id"],
                    "user2_folder": user2["folder"],
                    "file1": best_pair[0],
                    "file2": best_pair[1],
                    "score": round(best_score, 2)
                })
    
    # Sort results by score in descending order
    all_pairs_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "TestQA", "cross_user_matching_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_pairs_results, f, indent=2)
    
    logger.info(f"Analysis complete. Found {len(all_pairs_results)} pairs.")
    logger.info(f"Results saved to {output_path}")
    
    # Print top 10 matches for quick review
    print("\nTop 10 Potential False Positives (High scores between different users):")
    print("-" * 80)
    for i, res in enumerate(all_pairs_results[:20]):
        print(f"{i+1}. Score: {res['score']}%")
        print(f"   User 1: {res['user1_folder']} ({res['file1']})")
        print(f"   User 2: {res['user2_folder']} ({res['file2']})")
        print("-" * 80)

if __name__ == "__main__":
    run_cross_user_matching()
