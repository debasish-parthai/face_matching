import os
import json
from typing import List, Dict, Any

def combine_negative_scores():
    """
    Combine average scores from user1, user2, user3, and user4 avg_score.json files
    and create a summary file with overall statistics and descending score order.
    """
    users_dir = "Users"
    target_users = ["user1", "user2", "user3", "user4"]

    # Check if Users directory exists
    if not os.path.exists(users_dir):
        print(f"Directory '{users_dir}' not found!")
        return

    all_user_averages = []
    all_comparisons = []
    user_summaries = []

    # Process each target user
    for user_dir in target_users:
        user_path = os.path.join(users_dir, user_dir)
        false_negative_results_path = os.path.join(user_path, "false_negative_compare_results")
        avg_score_path = os.path.join(false_negative_results_path, "avg_score.json")

        # Check if avg_score.json exists for this user
        if not os.path.exists(avg_score_path):
            print(f"avg_score.json not found for {user_dir}")
            continue

        try:
            with open(avg_score_path, 'r') as f:
                user_data = json.load(f)

            # Collect user average for overall calculation
            user_avg = user_data.get("average_score", 0)
            all_user_averages.append(user_avg)

            # Create user summary
            user_summary = {
                "user": user_dir,
                "average_score": user_avg,
                "total_scores_count": user_data.get("total_scores_count", 0),
                "high_score_count": user_data.get("high_score_summary", {}).get("count", 0)
            }
            user_summaries.append(user_summary)

            # Collect all comparisons from this user
            comparisons = user_data.get("comparisons_summary", [])
            for comparison in comparisons:
                # Add user information to each comparison
                comparison_with_user = comparison.copy()
                comparison_with_user["source_user"] = user_dir
                all_comparisons.append(comparison_with_user)

            print(f"Processed {user_dir}: average_score={user_avg:.6f}, {len(comparisons)} comparisons")

        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error processing {user_dir}: {e}")
            continue

    if not all_user_averages:
        print("No valid data found from any users")
        return

    # Calculate overall average
    overall_average = sum(all_user_averages) / len(all_user_averages)

    # Sort all comparisons by score in descending order
    all_comparisons_sorted = sorted(all_comparisons, key=lambda x: x.get("score", 0), reverse=True)

    # Create summary statistics
    total_scores_all_users = sum(user["total_scores_count"] for user in user_summaries)
    total_high_scores = sum(user["high_score_count"] for user in user_summaries)

    # Create the final combined data structure
    combined_data = {
        "overall_summary": {
            "total_users_processed": len(user_summaries),
            "users_list": target_users,
            "overall_average_score": overall_average,
            "total_scores_across_all_users": total_scores_all_users,
            "total_high_scores_across_all_users": total_high_scores,
            "data_type": "combined_false_negative"
        },
        "user_summaries": user_summaries,
        "all_comparisons_sorted_desc": all_comparisons_sorted,
        "statistics": {
            "highest_score": all_comparisons_sorted[0]["score"] if all_comparisons_sorted else 0,
            "lowest_score": all_comparisons_sorted[-1]["score"] if all_comparisons_sorted else 0,
            "total_comparisons": len(all_comparisons_sorted),
            "scores_above_10": len([c for c in all_comparisons_sorted if c.get("score", 0) > 10]),
            "scores_above_15": len([c for c in all_comparisons_sorted if c.get("score", 0) > 15]),
            "scores_above_20": len([c for c in all_comparisons_sorted if c.get("score", 0) > 20])
        }
    }

    # Save to avg_negative_score.json in Testing folder
    output_path = "avg_negative_score.json"
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=4)

    print("\nCombined negative scores analysis completed!")
    print(f"Overall average score: {overall_average:.6f}")
    print(f"Total comparisons: {len(all_comparisons_sorted)}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    combine_negative_scores()