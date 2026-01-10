import os
import json
import glob

def calculate_avg_score():
    """
    Calculate average max_score for each user from their compare_results JSON files
    and save the result in avg_score.json
    """
    users_dir = "Users"

    # Check if Users directory exists
    if not os.path.exists(users_dir):
        print(f"Directory '{users_dir}' not found!")
        return

    # Get all user directories
    user_dirs = [d for d in os.listdir(users_dir) if os.path.isdir(os.path.join(users_dir, d)) and d.startswith("user")]

    for user_dir in user_dirs:
        user_path = os.path.join(users_dir, user_dir)
        compare_results_path = os.path.join(user_path, "compare_results")

        # Check if compare_results directory exists
        if not os.path.exists(compare_results_path):
            print(f"compare_results directory not found for {user_dir}")
            continue

        # Get all JSON files in compare_results (excluding avg_score.json if it exists)
        json_files = glob.glob(os.path.join(compare_results_path, "*.json"))
        json_files = [f for f in json_files if not os.path.basename(f) == "avg_score.json"]

        if not json_files:
            print(f"No JSON files found in {compare_results_path}")
            continue

        total_scores = []
        total_count = 0
        comparisons_summary = []

        # Process each JSON file
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                candidate_image_name = data.get("candidate_image_name", "unknown")

                # Extract max_score from reference_results
                if "reference_results" in data and data["reference_results"]:
                    for result in data["reference_results"]:
                        if "max_score" in result:
                            total_scores.append(result["max_score"])
                            total_count += 1

                            # Add to summary
                            comparison_summary = {
                                "candidate_image_name": candidate_image_name,
                                "comparing_with_image_name": result.get("comparing_with", "unknown"),
                                "score": result["max_score"],
                                "face_detected": result.get("faces_detected", 0)
                            }
                            comparisons_summary.append(comparison_summary)
                else:
                    # Handle case where reference_results is empty
                    comparison_summary = {
                        "candidate_image_name": candidate_image_name,
                        "comparing_with_image_name": "no_references",
                        "score": 0.0,
                        "face_detected": 0,
                        "note": "no_reference_results_found"
                    }
                    comparisons_summary.append(comparison_summary)

            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                print(f"Error processing {json_file}: {e}")
                continue

        # Filter out invalid comparisons (face_detected = 0 or score = 0.0)
        valid_comparisons = [comp for comp in comparisons_summary
                           if comp.get("face_detected", 0) > 0 and comp.get("score", 0.0) > 0.0]

        # Calculate average and create summary
        if total_count > 0:
            average_score = sum(total_scores) / total_count

            # Save to avg_score.json
            avg_score_data = {
                "average_score": average_score,
                "total_scores_count": total_count,
                "user": user_dir,
                "comparisons_summary": valid_comparisons
            }

            avg_score_path = os.path.join(compare_results_path, "avg_score.json")
            with open(avg_score_path, 'w') as f:
                json.dump(avg_score_data, f, indent=4)

            print(f"Calculated average score for {user_dir}: {average_score:.6f} (from {total_count} scores)")
        else:
            # Handle case where no valid scores were found
            avg_score_data = {
                "average_score": 0.0,
                "total_scores_count": 0,
                "user": user_dir,
                "comparisons_summary": valid_comparisons,
                "note": "no_valid_scores_found"
            }

            avg_score_path = os.path.join(compare_results_path, "avg_score.json")
            with open(avg_score_path, 'w') as f:
                json.dump(avg_score_data, f, indent=4)

            print(f"No valid scores found for {user_dir}, saved summary with {len(valid_comparisons)} entries")

if __name__ == "__main__":
    calculate_avg_score()
