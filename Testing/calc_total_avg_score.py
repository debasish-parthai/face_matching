import os
import json

def calculate_total_avg_score():
    """
    Calculate the total average score across all users from their avg_score.json files
    and save the result in avg_score.json in the Users folder
    """
    users_dir = "Users"

    # Check if Users directory exists
    if not os.path.exists(users_dir):
        print(f"Directory '{users_dir}' not found!")
        return

    # Get all user directories (user1 to user25)
    user_dirs = [f"user{i}" for i in range(1, 26)]

    total_average_scores = []
    processed_users = 0
    problematic_comparisons = []
    users_summary = []

    for user_dir in user_dirs:
        user_path = os.path.join(users_dir, user_dir)
        compare_results_path = os.path.join(user_path, "compare_results")
        avg_score_file = os.path.join(compare_results_path, "avg_score.json")

        # Check if avg_score.json exists for this user
        if os.path.exists(avg_score_file):
            try:
                with open(avg_score_file, 'r') as f:
                    data = json.load(f)

                if "average_score" in data:
                    total_average_scores.append(data["average_score"])
                    processed_users += 1
                    print(f"Processed {user_dir}: {data['average_score']:.6f}")

                    # Add user summary
                    user_summary = {
                        "user": user_dir,
                        "average_score": data["average_score"],
                        "total_scores_count": data.get("total_scores_count", 0)
                    }
                    users_summary.append(user_summary)
                else:
                    print(f"Warning: average_score not found in {avg_score_file}")

                # Scan comparisons_summary for problematic entries
                if "comparisons_summary" in data:
                    for comparison in data["comparisons_summary"]:
                        # Check for problematic conditions
                        has_zero_score = comparison.get("score", 0) == 0.0
                        has_zero_faces = comparison.get("face_detected", 0) == 0
                        has_no_reference_note = comparison.get("note") == "no_reference_results_found"

                        if has_zero_score or has_zero_faces or has_no_reference_note:
                            # Add user information to the comparison
                            problematic_comparison = comparison.copy()
                            problematic_comparison["user"] = user_dir
                            problematic_comparisons.append(problematic_comparison)

            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                print(f"Error processing {avg_score_file}: {e}")
                continue
        else:
            print(f"Warning: avg_score.json not found for {user_dir}")

    # Calculate total average
    if processed_users > 0:
        total_avg_score = sum(total_average_scores) / processed_users

        # Save to avg_score.json in Users folder
        total_avg_score_data = {
            "total_average_score": total_avg_score,
            "total_users_processed": processed_users,
            "total_users_expected": 25,
            "users_summary": users_summary,
            "problematic_comparisons_summary": problematic_comparisons,
            "total_problematic_comparisons": len(problematic_comparisons)
        }

        total_avg_score_path = os.path.join(users_dir, "avg_score.json")
        with open(total_avg_score_path, 'w') as f:
            json.dump(total_avg_score_data, f, indent=4)

        print(f"\nCalculated total average score: {total_avg_score:.6f}")
        print(f"Processed {processed_users} out of 25 users")
        print(f"Found {len(problematic_comparisons)} problematic comparisons")
        print(f"Saved to {total_avg_score_path}")
    else:
        # Save empty summary if no users processed
        total_avg_score_data = {
            "total_average_score": 0.0,
            "total_users_processed": 0,
            "total_users_expected": 25,
            "users_summary": users_summary,
            "problematic_comparisons_summary": problematic_comparisons,
            "total_problematic_comparisons": len(problematic_comparisons)
        }

        total_avg_score_path = os.path.join(users_dir, "avg_score.json")
        with open(total_avg_score_path, 'w') as f:
            json.dump(total_avg_score_data, f, indent=4)

        print("No avg_score.json files found to process!")
        print(f"Saved problematic comparisons summary to {total_avg_score_path}")

if __name__ == "__main__":
    calculate_total_avg_score()
