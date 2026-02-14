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
    """Save results categorized by score ranges, splitting large files into parts and cleaning up old versions"""
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
        single_filepath = os.path.join(output_dir, f"{range_name}.json")

        # If more than 100 records, split into parts
        if num_results > 100:
            chunk_size = 100
            num_parts = (num_results + chunk_size - 1) // chunk_size

            # Delete the single file if it exists to prevent double-loading on resume
            if os.path.exists(single_filepath):
                try:
                    os.remove(single_filepath)
                except Exception:
                    pass

            for part in range(num_parts):
                start_idx = part * chunk_size
                end_idx = min((part + 1) * chunk_size, num_results)
                part_results = results[start_idx:end_idx]

                filename = f"{range_name}_part{part+1}.json"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(part_results, f, indent=2)

            # Clean up any leftover old parts
            part_num = num_parts + 1
            while True:
                old_part = os.path.join(output_dir, f"{range_name}_part{part_num}.json")
                if os.path.exists(old_part):
                    try:
                        os.remove(old_part)
                        part_num += 1
                    except Exception:
                        break
                else:
                    break
        else:
            # Single file for this range
            with open(single_filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

            # Cleanup any existing part files for this range
            part_num = 1
            while True:
                part_file = os.path.join(output_dir, f"{range_name}_part{part_num}.json")
                if os.path.exists(part_file):
                    try:
                        os.remove(part_file)
                        part_num += 1
                    except Exception:
                        break
                else:
                    break

def save_progress_state(progress_file: str, current_i: int, current_j: int, processed_pairs: int):
    """Save current processing progress to resume later"""
    progress_data = {
        "current_i": current_i,
        "current_j": current_j,
        "processed_pairs": processed_pairs,
        "timestamp": datetime.now().isoformat()
    }

    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2)
        logger.info(f"Progress saved: user {current_i+1}, comparison {current_j+1}, {processed_pairs} pairs processed")
    except Exception as e:
        logger.error(f"Failed to save progress: {str(e)}")

def load_progress_state(progress_file: str) -> Tuple[int, int, int]:
    """Load previous processing progress. Returns (current_i, current_j, processed_pairs)"""
    if not os.path.exists(progress_file):
        return 0, 0, 0

    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)

        current_i = progress_data.get("current_i", 0)
        current_j = progress_data.get("current_j", 0)
        processed_pairs = progress_data.get("processed_pairs", 0)

        logger.info(f"Resuming from: user {current_i+1}, comparison {current_j+1}, {processed_pairs} pairs processed")
        return current_i, current_j, processed_pairs
    except Exception as e:
        logger.error(f"Failed to load progress: {str(e)}")
        return 0, 0, 0

def save_individual_comparisons(user1_folder: str, user2_folder: str, comparisons: List[Dict], output_dir: str):
    """Save all individual image comparisons between two users to a JSON file"""
    if not comparisons:
        return

    filename = f"{user1_folder}_{user2_folder}.json"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparisons, f, indent=2)
        logger.info(f"Saved {len(comparisons)} comparisons to {filename}")
    except Exception as e:
        logger.error(f"Failed to save {filename}: {str(e)}")

def save_user_summary(user1_folder: str, user_comparisons: List[Dict], output_dir: str):
    """Save all >40 score comparisons for a user"""
    if not user_comparisons:
        return

    filename = f"{user1_folder}_summary.json"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(user_comparisons, f, indent=2)
        logger.info(f"Saved user summary {filename} with {len(user_comparisons)} comparisons")
    except Exception as e:
        logger.error(f"Failed to save {filename}: {str(e)}")

def get_latest_part_file(base_path: str) -> Tuple[str, int]:
    """Find the latest part file and its index"""
    dir_name = os.path.dirname(base_path)
    file_name = os.path.basename(base_path)
    name_part, ext = os.path.splitext(file_name)
    
    parts = []
    if os.path.exists(dir_name):
        for f in os.listdir(dir_name):
            if f.startswith(name_part + "_part") and f.endswith(ext):
                try:
                    idx = int(f.split("_part")[-1].split(ext)[0])
                    parts.append(idx)
                except ValueError: continue
    
    if not parts:
        return f"{name_part}_part1{ext}", 1
    
    latest_idx = max(parts)
    return f"{name_part}_part{latest_idx}{ext}", latest_idx

def save_master_chunk(new_data: List[Dict], base_path: str, max_size_mb: int = 15):
    """Append data to the current part file, or start a new one if it exceeds max_size_mb"""
    if not new_data:
        return

    latest_file, latest_idx = get_latest_part_file(base_path)
    latest_path = os.path.join(os.path.dirname(base_path), latest_file)
    
    existing_data = []
    if os.path.exists(latest_path):
        # Check size
        if os.path.getsize(latest_path) > max_size_mb * 1024 * 1024:
            # Start new part
            latest_idx += 1
            latest_file = f"{os.path.basename(base_path).split('.')[0]}_part{latest_idx}.json"
            latest_path = os.path.join(os.path.dirname(base_path), latest_file)
        else:
            try:
                with open(latest_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = []

    existing_data.extend(new_data)
    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2)

def run_cross_user_matching_resumable():
    users_dir = os.path.join(os.path.dirname(__file__), "TestQA", "Users_Final")
    master_best_base = os.path.join(os.path.dirname(__file__), "TestQA", "master_best_matches.json")
    master_sig_base = os.path.join(os.path.dirname(__file__), "TestQA", "master_significant_matches.json")
    progress_file = os.path.join(os.path.dirname(__file__), "TestQA", "progress_state.json")
    categorized_dir = os.path.join(os.path.dirname(__file__), "TestQA", "categorized_similarity_results")
    summary_dir = os.path.join(os.path.dirname(__file__), "TestQA", "user_summaries")

    os.makedirs(categorized_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    logger.info("Starting Phase 1: Chunked matching analysis (15MB per master part)...")
    users = load_user_data(users_dir)
    num_users = len(users)
    start_i, start_j, processed_pairs = load_progress_state(progress_file)

    # Buffers for current session matches (flushed and cleared periodically)
    session_best_buffer = []
    session_sig_buffer = []

    for i in range(start_i, num_users):
        user1 = users[i]
        user1_folder = user1["folder"]

        if i % 10 == 0:
            logger.info(f"Processing user {i+1}/{num_users}...")

        j_start = max(start_j + 1, i + 1) if i == start_i else i + 1

        for j in range(j_start, num_users):
            user2 = users[j]
            user2_folder = user2["folder"]

            best_score = -1.0
            best_pair_data = None

            for emb1_data in user1["embeddings"]:
                for emb2_data in user2["embeddings"]:
                    score = calculate_similarity(emb1_data["embedding"], emb2_data["embedding"])
                    
                    match_data = {
                        "user1_id": user1["user_id"], "user1_folder": user1_folder,
                        "user2_id": user2["user_id"], "user2_folder": user2_folder,
                        "file1": emb1_data["file_id"], "file2": emb2_data["file_id"],
                        "score": round(score, 2)
                    }

                    if score >= 40:
                        session_sig_buffer.append(match_data)
                    if score > best_score:
                        best_score = score
                        best_pair_data = match_data

            if best_pair_data:
                session_best_buffer.append(best_pair_data)

            processed_pairs += 1

            # Flush buffers to disk and CLEAR memory
            if processed_pairs % 500 == 0 or j == num_users - 1:
                save_progress_state(progress_file, i, j, processed_pairs)
                save_master_chunk(session_best_buffer, master_best_base)
                save_master_chunk(session_sig_buffer, master_sig_base)
                
                # CRITICAL: Clear memory after saving to disk
                session_best_buffer = []
                session_sig_buffer = []

        start_j = i + 1

    if os.path.exists(progress_file): os.remove(progress_file)
    logger.info("Phase 1 Complete. Triggering Phase 2 Summarization...")
    summarize_matching_results(master_best_base, master_sig_base, categorized_dir, summary_dir)

def summarize_matching_results(best_base: str, sig_base: str, categorized_dir: str, summary_dir: str):
    """Phase 2: Load all master parts and summarize"""
    logger.info("Starting Phase 2: Summarizing results from all master parts...")
    
    def load_all_parts(base_path):
        data = []
        dir_name = os.path.dirname(base_path)
        base_name = os.path.basename(base_path).split(".")[0]
        if not os.path.exists(dir_name): return data
        
        for f in sorted(os.listdir(dir_name)):
            if f.startswith(base_name + "_part") and f.endswith(".json"):
                try:
                    with open(os.path.join(dir_name, f), 'r', encoding='utf-8') as file:
                        data.extend(json.load(file))
                    logger.info(f"Loaded {f}")
                except Exception as e: logger.error(f"Error loading {f}: {e}")
        return data

    # 1. Categorize Best Matches
    best_matches = load_all_parts(best_base)
    if best_matches:
        results_by_range = {k: [] for k in ["above_90", "80_90", "70_80", "60_70", "50_60", "40_50", "30_40", "20_30", "0_20"]}
        for m in best_matches:
            results_by_range[get_score_range(m["score"])].append(m)
        for k in results_by_range:
            results_by_range[k].sort(key=lambda x: x["score"], reverse=True)
        save_categorized_results(results_by_range, categorized_dir)

    # 2. Generate User Summaries
    sig_matches = load_all_parts(sig_base)
    if sig_matches:
        user_summaries = {}
        for m in sig_matches:
            for u in [m["user1_folder"], m["user2_folder"]]:
                if u not in user_summaries: user_summaries[u] = []
                user_summaries[u].append(m)
        
        for folder, matches in user_summaries.items():
            matches.sort(key=lambda x: x["score"], reverse=True)
            with open(os.path.join(summary_dir, f"{folder}_summary.json"), 'w', encoding='utf-8') as f:
                json.dump(matches, f, indent=2)

    logger.info("Phase 2 Complete. All results summarized.")


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
        # Run the resumable cross-user matching analysis
        run_cross_user_matching_resumable()
