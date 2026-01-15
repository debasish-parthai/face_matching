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
    user_folders.sort(key=lambda x: int(x.split('user')[-1]))
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

def get_score_range(score: float) -> str:
    """Determine which score range category a score belongs to"""
    if score > 90:
        return "above_90"
    elif 80 <= score <= 90:
        return "80_90"
    elif 70 <= score < 80:
        return "70_80"
    elif 60 <= score < 70:
        return "60_70"
    elif 50 <= score < 60:
        return "50_60"
    elif 40 <= score < 50:
        return "40_50"
    elif 30 <= score < 40:
        return "30_40"
    elif 20 <= score < 30:
        return "20_30"
    else:  # 0-20
        return "0_20"

def save_categorized_results(results_by_range: Dict[str, List[Dict]], output_dir: str):
    """Save results categorized by score ranges, splitting large files into parts"""
    range_names = {
        "above_90": "above_90",
        "80_90": "80_90",
        "70_80": "70_80",
        "60_70": "60_70",
        "50_60": "50_60",
        "40_50": "40_50",
        "30_40": "30_40",
        "20_30": "20_30",
        "0_20": "0_20"
    }

    for range_key, results in results_by_range.items():
        if not results:
            continue

        range_name = range_names[range_key]
        num_results = len(results)

        # If more than 100 records, split into parts
        if num_results > 100:
            chunk_size = 100
            num_parts = (num_results + chunk_size - 1) // chunk_size  # Ceiling division

            for part in range(num_parts):
                start_idx = part * chunk_size
                end_idx = min((part + 1) * chunk_size, num_results)
                part_results = results[start_idx:end_idx]

                filename = f"{range_name}_part{part+1}.json"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(part_results, f, indent=2)

                logger.info(f"Saved {len(part_results)} results to {filename} (range: {range_key})")
        else:
            # Single file for this range
            filename = f"{range_name}.json"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

            logger.info(f"Saved {len(results)} results to {filename} (range: {range_key})")

def run_cross_user_matching():
    users_dir = os.path.join(os.path.dirname(__file__), "TestQA", "Users_Final")
    output_dir = os.path.join(os.path.dirname(__file__), "TestQA", "categorized_similarity_results")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Starting cross-user matching analysis on {users_dir}...")
    logger.info(f"Results will be saved to {output_dir}")

    users = load_user_data(users_dir)
    num_users = len(users)
    logger.info(f"Loaded {num_users} users with embeddings.")

    # Initialize results dictionary for each score range
    results_by_range = {
        "above_90": [],
        "80_90": [],
        "70_80": [],
        "60_70": [],
        "50_60": [],
        "40_50": [],
        "30_40": [],
        "20_30": [],
        "0_20": []
    }

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
            comparison_count = 0
            for emb1_data in user1["embeddings"]:
                for emb2_data in user2["embeddings"]:
                    score = calculate_similarity(emb1_data["embedding"], emb2_data["embedding"])
                    comparison_count += 1

                    # Set best_pair to first comparison if not set
                    if best_pair is None:
                        best_score = score
                        best_pair = (emb1_data["file_id"], emb2_data["file_id"])
                    elif score > best_score:
                        best_score = score
                        best_pair = (emb1_data["file_id"], emb2_data["file_id"])

          
            result = {
                "user1_id": user1["user_id"],
                "user1_folder": user1["folder"],
                "user2_id": user2["user_id"],
                "user2_folder": user2["folder"],
                "file1": best_pair[0],
                "file2": best_pair[1],
                "score": round(best_score, 2)
            }

            # Categorize result by score range
            range_key = get_score_range(best_score)
            results_by_range[range_key].append(result)

    # Sort each range by score in descending order
    for range_key in results_by_range:
        results_by_range[range_key].sort(key=lambda x: x["score"], reverse=True)

    # Save categorized results
    save_categorized_results(results_by_range, output_dir)

    # Count total results
    total_results = sum(len(results) for results in results_by_range.values())
    logger.info(f"Analysis complete. Found {total_results} pairs across all score ranges.")

    # Print summary
    print("\nSummary of results by score range:")
    print("-" * 50)
    range_names_display = {
        "above_90": "Above 90",
        "80_90": "80-90",
        "70_80": "70-80",
        "60_70": "60-70",
        "50_60": "50-60",
        "40_50": "40-50",
        "30_40": "30-40",
        "20_30": "20-30",
        "0_20": "0-20"
    }

    for range_key, results in results_by_range.items():
        if results:
            range_name = range_names_display[range_key]
            print(f"{range_name}: {len(results)} pairs")
            if len(results) > 100:
                parts = (len(results) + 99) // 100  # Ceiling division by 100
                print(f"  Split into {parts} parts")

    # Print top matches for quick review
    print("\nTop 10 Potential False Positives (High scores between different users):")
    print("-" * 80)
    top_results = []
    for results in results_by_range.values():
        top_results.extend(results[:10])  # Take top 10 from each range
    top_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = top_results[:20]  # Take overall top 20

    for i, res in enumerate(top_results):
        print(f"{i+1}. Score: {res['score']}%")
        print(f"   User 1: {res['user1_folder']} ({res['file1']})")
        print(f"   User 2: {res['user2_folder']} ({res['file2']})")
        print("-" * 80)

def chunk_cross_user_matching_results(chunk_size: int = 5000):
    """Chunk the cross_user_matching_results.json file into smaller files of specified size"""
    input_path = os.path.join(os.path.dirname(__file__), "TestQA", "cross_user_matching_results.json")

    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading large results file: {input_path}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file: {str(e)}")
        return

    total_results = len(all_results)
    logger.info(f"Loaded {total_results} results. Chunking into groups of {chunk_size}...")

    # Calculate number of chunks needed
    num_chunks = (total_results + chunk_size - 1) // chunk_size  # Ceiling division

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_results)

        chunk_data = all_results[start_idx:end_idx]

        output_filename = f"cross_user_matching_results{i+1}.json"
        output_path = os.path.join(os.path.dirname(__file__), "TestQA", output_filename)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, indent=2)

            logger.info(f"Created chunk {i+1}/{num_chunks}: {output_filename} ({len(chunk_data)} items)")

        except Exception as e:
            logger.error(f"Error saving chunk {i+1}: {str(e)}")

    logger.info(f"Chunking complete. Created {num_chunks} files.")

def test_categorization_logic():
    """Test the score categorization and file splitting logic"""
    import tempfile
    import shutil

    # Create sample results for testing - focus on ranges that will exceed 100 records
    sample_results = []

    # Create many results in 80-90 range to test splitting
    for i in range(250):  # Create 250 sample results
        score = 85 + (i * 0.04) % 5  # Scores mostly in 80-90 range
        sample_results.append({
            "user1_id": f"user{i//10 + 1}",
            "user1_folder": f"user{i//10 + 1}",
            "user2_id": f"user{(i//10 + 2) % 25 + 1}",
            "user2_folder": f"user{(i//10 + 2) % 25 + 1}",
            "file1": f"user{i//10 + 1}_01.jpg",
            "file2": f"user{(i//10 + 2) % 25 + 1}_01.jpg",
            "score": round(score, 2)
        })

    # Test categorization
    results_by_range = {
        "above_90": [],
        "80_90": [],
        "70_80": [],
        "60_70": [],
        "50_60": [],
        "40_50": [],
        "30_40": [],
        "20_30": [],
        "0_20": []
    }

    for result in sample_results:
        range_key = get_score_range(result["score"])
        results_by_range[range_key].append(result)

    # Sort each range
    for range_key in results_by_range:
        results_by_range[range_key].sort(key=lambda x: x["score"], reverse=True)

    print("Results distribution before saving:")
    for range_key, results in results_by_range.items():
        if results:
            print(f"  {range_key}: {len(results)} records")

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        save_categorized_results(results_by_range, temp_dir)

        # Check created files
        files = os.listdir(temp_dir)
        print(f"\nTest created {len(files)} files:")
        for file in sorted(files):
            filepath = os.path.join(temp_dir, file)
            with open(filepath, 'r') as f:
                data = json.load(f)
            print(f"  {file}: {len(data)} records")

    print("Categorization and file splitting test completed successfully!")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "chunk":
        # Run chunking function
        chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
        chunk_cross_user_matching_results(chunk_size)
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test function
        test_categorization_logic()
    else:
        # Run the original cross-user matching analysis
        run_cross_user_matching()
