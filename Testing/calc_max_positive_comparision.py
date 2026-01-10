import os
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaxScoreCalculator:
    def __init__(self, users_base_folder: str):
        self.users_base_folder = users_base_folder

    def calculate_max_min_scores(self, comparisons_summary: list) -> dict:
        """Calculate max and min scores from comparisons_summary array"""
        if not comparisons_summary:
            return None

        # Find max score
        max_comparison = max(comparisons_summary, key=lambda x: x['score'])
        max_score_data = {
            "candidate_image_name": max_comparison["candidate_image_name"],
            "comparing_with_image_name": max_comparison["comparing_with_image_name"],
            "score": max_comparison["score"],
            "face_detected": max_comparison["face_detected"]
        }

        # Find min score
        min_comparison = min(comparisons_summary, key=lambda x: x['score'])
        min_score_data = {
            "candidate_image_name": min_comparison["candidate_image_name"],
            "comparing_with_image_name": min_comparison["comparing_with_image_name"],
            "score": min_comparison["score"],
            "face_detected": min_comparison["face_detected"]
        }

        return {
            "max_score": max_score_data,
            "min_score": min_score_data
        }

    def process_user_folder(self, user_folder: str):
        """Process a single user folder to calculate max/min scores"""
        user_name = os.path.basename(user_folder)
        logger.info(f"Processing user: {user_name}")

        # Path to avg_score.json
        avg_score_path = os.path.join(user_folder, "compare_results", "avg_score.json")

        if not os.path.exists(avg_score_path):
            logger.warning(f"avg_score.json not found for {user_name}")
            return

        try:
            # Read avg_score.json
            with open(avg_score_path, 'r') as f:
                data = json.load(f)

            comparisons_summary = data.get('comparisons_summary', [])

            if not comparisons_summary:
                logger.warning(f"No comparisons_summary found for {user_name}")
                return

            # Calculate max and min scores
            result = self.calculate_max_min_scores(comparisons_summary)

            if result:
                # Path to save max_score.json
                max_score_path = os.path.join(user_folder, "compare_results", "max_score.json")

                # Save max_score.json
                with open(max_score_path, 'w') as f:
                    json.dump(result, f, indent=2)

                logger.info(f"Saved max_score.json for {user_name}: max={result['max_score']['score']:.2f}, min={result['min_score']['score']:.2f}")
            else:
                logger.error(f"Failed to calculate scores for {user_name}")

        except Exception as e:
            logger.error(f"Error processing {user_name}: {e}")

    def process_all_users(self):
        """Process all user folders"""
        if not os.path.exists(self.users_base_folder):
            logger.error(f"Users folder not found: {self.users_base_folder}")
            return

        # Get all user folders (user1, user2, ..., user25)
        user_folders = []
        for item in os.listdir(self.users_base_folder):
            item_path = os.path.join(self.users_base_folder, item)
            if os.path.isdir(item_path) and item.startswith('user'):
                user_folders.append(item_path)

        user_folders.sort()

        logger.info(f"Found {len(user_folders)} user folders")

        for user_folder in user_folders:
            try:
                self.process_user_folder(user_folder)
            except Exception as e:
                logger.error(f"Error processing user folder {user_folder}: {e}")

def main():
    # Path to Users folder
    users_folder = os.path.join(os.path.dirname(__file__), "Users")

    # Create calculator instance
    calculator = MaxScoreCalculator(users_folder)

    # Process all users
    calculator.process_all_users()

    logger.info("Max score calculation completed for all users!")

if __name__ == "__main__":
    main()
