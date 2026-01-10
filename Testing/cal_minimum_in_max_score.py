import os
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinMaxScoreAnalyzer:
    def __init__(self, users_base_folder: str):
        self.users_base_folder = users_base_folder

    def analyze_all_max_scores(self):
        """Analyze all max_score.json files across all users"""
        if not os.path.exists(self.users_base_folder):
            logger.error(f"Users folder not found: {self.users_base_folder}")
            return None

        # Collections for analysis
        max_scores_data = []  # All max_score objects from all users
        min_scores_data = []  # All min_score objects from all users

        # Get all user folders
        user_folders = []
        for item in os.listdir(self.users_base_folder):
            item_path = os.path.join(self.users_base_folder, item)
            if os.path.isdir(item_path) and item.startswith('user'):
                user_folders.append(item_path)

        user_folders.sort()

        logger.info(f"Found {len(user_folders)} user folders")

        # Process each user
        for user_folder in user_folders:
            user_name = os.path.basename(user_folder)
            max_score_path = os.path.join(user_folder, "compare_results", "max_score.json")

            if not os.path.exists(max_score_path):
                logger.warning(f"max_score.json not found for {user_name}")
                continue

            try:
                with open(max_score_path, 'r') as f:
                    data = json.load(f)

                if 'max_score' in data:
                    max_scores_data.append(data['max_score'])
                if 'min_score' in data:
                    min_scores_data.append(data['min_score'])

                logger.info(f"Processed {user_name}")

            except Exception as e:
                logger.error(f"Error processing {user_name}: {e}")

        if not max_scores_data or not min_scores_data:
            logger.error("No valid data found")
            return None

        # Find min and max among all max_scores
        min_among_max_scores = min(max_scores_data, key=lambda x: x['score'])
        max_among_max_scores = max(max_scores_data, key=lambda x: x['score'])

        # Find min and max among all min_scores
        min_among_min_scores = min(min_scores_data, key=lambda x: x['score'])
        max_among_min_scores = max(min_scores_data, key=lambda x: x['score'])

        # Sort collections in descending order by score
        max_scores_descending = sorted(max_scores_data, key=lambda x: x['score'], reverse=True)
        min_scores_descending = sorted(min_scores_data, key=lambda x: x['score'], reverse=True)

        # Create result structure
        result = {
            "analysis_summary": {
                "total_users_processed": len([d for d in user_folders if os.path.exists(os.path.join(d, "compare_results", "max_score.json"))]),
                "description": "Analysis of max_score.json files across all users"
            },
            "min_among_all_max_scores": {
                "candidate_image_name": min_among_max_scores["candidate_image_name"],
                "comparing_with_image_name": min_among_max_scores["comparing_with_image_name"],
                "score": min_among_max_scores["score"],
                "face_detected": min_among_max_scores["face_detected"]
            },
            "max_among_all_max_scores": {
                "candidate_image_name": max_among_max_scores["candidate_image_name"],
                "comparing_with_image_name": max_among_max_scores["comparing_with_image_name"],
                "score": max_among_max_scores["score"],
                "face_detected": max_among_max_scores["face_detected"]
            },
            "min_among_all_min_scores": {
                "candidate_image_name": min_among_min_scores["candidate_image_name"],
                "comparing_with_image_name": min_among_min_scores["comparing_with_image_name"],
                "score": min_among_min_scores["score"],
                "face_detected": min_among_min_scores["face_detected"]
            },
            "max_among_all_min_scores": {
                "candidate_image_name": max_among_min_scores["candidate_image_name"],
                "comparing_with_image_name": max_among_min_scores["comparing_with_image_name"],
                "score": max_among_min_scores["score"],
                "face_detected": max_among_min_scores["face_detected"]
            },
            "all_max_scores_descending": max_scores_descending,
            "all_min_scores_descending": min_scores_descending
        }

        logger.info(f"Analysis complete:")
        logger.info(f"  Min among all max_scores: {result['min_among_all_max_scores']['score']:.2f}")
        logger.info(f"  Max among all max_scores: {result['max_among_all_max_scores']['score']:.2f}")
        logger.info(f"  Min among all min_scores: {result['min_among_all_min_scores']['score']:.2f}")
        logger.info(f"  Max among all min_scores: {result['max_among_all_min_scores']['score']:.2f}")

        return result

    def save_analysis_result(self, result: dict, output_path: str):
        """Save analysis result to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved analysis result to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving result to {output_path}: {e}")

def main():
    # Path to Users folder
    users_folder = os.path.join(os.path.dirname(__file__), "Users")

    # Create analyzer instance
    analyzer = MinMaxScoreAnalyzer(users_folder)

    # Analyze all max_score.json files
    result = analyzer.analyze_all_max_scores()

    if result:
        # Save result to Testing folder
        output_path = os.path.join(os.path.dirname(__file__), "min_max_analysis_result.json")
        analyzer.save_analysis_result(result, output_path)
        logger.info("Analysis completed successfully!")
    else:
        logger.error("Analysis failed!")

if __name__ == "__main__":
    main()
