import os
import json
import glob

def calculate_negative_avg_score():
    """
    Calculate average max_score for user1 from their false_negative_compare_results JSON files
    and save the result in avg_score.json in the false_negative_compare_results folder.
    Also includes summary of scores greater than 5.
    """
    users_dir = "Users"
    user_dir = "user1"  # Only for user1

    # Check if Users directory exists
    if not os.path.exists(users_dir):
        print(f"Directory '{users_dir}' not found!")
        return

    user_path = os.path.join(users_dir, user_dir)
    false_negative_results_path = os.path.join(user_path, "false_negative_compare_results")

    # Check if false_negative_compare_results directory exists
    if not os.path.exists(false_negative_results_path):
        print(f"false_negative_compare_results directory not found for {user_dir}")
        return

    # Get all JSON files in false_negative_compare_results (excluding avg_score.json if it exists)
    json_files = glob.glob(os.path.join(false_negative_results_path, "*.json"))
    json_files = [f for f in json_files if not os.path.basename(f) == "avg_score.json"]

    if not json_files:
        print(f"No JSON files found in {false_negative_results_path}")
        return

    total_scores = []
    total_count = 0
    comparisons_summary = []
    high_score_summary = []  # For scores > 5

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
                        score = result["max_score"]
                        total_scores.append(score)
                        total_count += 1

                        # Add to main summary
                        comparison_summary = {
                            "candidate_image_name": candidate_image_name,
                            "comparing_with_image_name": result.get("comparing_with", "unknown"),
                            "score": score,
                            "face_detected": result.get("faces_detected", 0),
                            "compared_user": result.get("compared_user", "unknown")
                        }
                        comparisons_summary.append(comparison_summary)

                        # Add to high score summary if score > 5
                        if score > 5:
                            high_score_summary.append(comparison_summary.copy())
            else:
                # Handle case where reference_results is empty
                comparison_summary = {
                    "candidate_image_name": candidate_image_name,
                    "comparing_with_image_name": "no_references",
                    "score": 0.0,
                    "face_detected": 0,
                    "compared_user": "unknown",
                    "note": "no_reference_results_found"
                }
                comparisons_summary.append(comparison_summary)

        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error processing {json_file}: {e}")
            continue

    # Calculate average and create summary
    if total_count > 0:
        average_score = sum(total_scores) / total_count

        # Save to avg_score.json
        avg_score_data = {
            "average_score": average_score,
            "total_scores_count": total_count,
            "user": user_dir,
            "data_type": "false_negative",
            "comparisons_summary": comparisons_summary,
            "high_score_summary": {
                "count": len(high_score_summary),
                "threshold": 5.0,
                "description": "Comparisons with max_score greater than 5",
                "results": high_score_summary
            }
        }

        avg_score_path = os.path.join(false_negative_results_path, "avg_score.json")
        with open(avg_score_path, 'w') as f:
            json.dump(avg_score_data, f, indent=4)

        print(f"Calculated average score for {user_dir} false_negative results: {average_score:.6f} (from {total_count} scores)")
        print(f"Found {len(high_score_summary)} comparisons with score > 5")
    else:
        # Handle case where no valid scores were found
        avg_score_data = {
            "average_score": 0.0,
            "total_scores_count": 0,
            "user": user_dir,
            "data_type": "false_negative",
            "comparisons_summary": comparisons_summary,
            "high_score_summary": {
                "count": 0,
                "threshold": 5.0,
                "description": "Comparisons with max_score greater than 5",
                "results": []
            },
            "note": "no_valid_scores_found"
        }

        avg_score_path = os.path.join(false_negative_results_path, "avg_score.json")
        with open(avg_score_path, 'w') as f:
            json.dump(avg_score_data, f, indent=4)

        print(f"No valid scores found for {user_dir} false_negative results, saved summary with {len(comparisons_summary)} entries")

if __name__ == "__main__":
    calculate_negative_avg_score()
