import os
import requests
import json
from pathlib import Path
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceMatchingTester:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.match_endpoint = f"{api_url}/match-faces/"

    def get_image_files(self, user_folder: str) -> List[str]:
        """Get all JPG image files from a user folder"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []

        for file in os.listdir(user_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(user_folder, file))

        return sorted(image_files)

    def call_match_faces_api(self, candidate_path: str, reference_paths: List[str]) -> dict:
        """Call the match-faces API with candidate and reference images"""
        # Keep file handles open until after the request
        file_handles = []

        try:
            # Prepare files as list of tuples for multipart/form-data
            files = []

            # Add candidate file
            candidate_handle = open(candidate_path, 'rb')
            file_handles.append(candidate_handle)
            files.append(('candidate_file', ('candidate.jpg', candidate_handle, 'image/jpeg')))

            # Add reference files (multiple files with same field name)
            for i, ref_path in enumerate(reference_paths):
                ref_handle = open(ref_path, 'rb')
                file_handles.append(ref_handle)
                files.append(('reference_files', (f'reference_{i}.jpg', ref_handle, 'image/jpeg')))

            # Make API request
            response = requests.post(self.match_endpoint, files=files)
            response.raise_for_status()

            result = response.json()

            # Add candidate image name
            candidate_filename = os.path.basename(candidate_path)
            result['candidate_image_name'] = candidate_filename

            # Add comparing_with field to each reference result
            for i, ref_path in enumerate(reference_paths):
                if i < len(result['reference_results']):
                    comparing_filename = os.path.basename(ref_path)
                    result['reference_results'][i]['comparing_with'] = comparing_filename

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error calling API: {e}")
            return None
        finally:
            # Always close file handles
            for handle in file_handles:
                try:
                    handle.close()
                except:
                    pass

    def save_comparison_result(self, result: dict, output_path: str):
        """Save comparison result to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved result to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving result to {output_path}: {e}")

    def test_user_folder(self, user_folder: str):
        """Test all pairwise comparisons within a user folder"""
        user_name = os.path.basename(user_folder)
        logger.info(f"Testing user: {user_name}")

        # Get all image files
        image_files = self.get_image_files(user_folder)
        if len(image_files) < 2:
            logger.warning(f"User {user_name} has less than 2 images, skipping")
            return

        logger.info(f"Found {len(image_files)} images for {user_name}")

        # Create compare_results folder
        results_folder = os.path.join(user_folder, "compare_results")
        os.makedirs(results_folder, exist_ok=True)

        comparison_count = 0

        # Perform pairwise comparisons
        for i, candidate_path in enumerate(image_files):
            # References are all images after the current candidate
            reference_paths = image_files[i+1:]

            if not reference_paths:
                continue

            candidate_filename = os.path.basename(candidate_path)
            logger.info(f"Comparing {candidate_filename} with {len(reference_paths)} reference images")

            # Call API
            result = self.call_match_faces_api(candidate_path, reference_paths)

            if result:
                # Save result
                result_filename = f"comparison_{i+1:02d}_{candidate_filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('.bmp', '')}.json"
                result_path = os.path.join(results_folder, result_filename)
                self.save_comparison_result(result, result_path)
                comparison_count += 1
            else:
                logger.error(f"Failed to get result for {candidate_filename}")

        logger.info(f"Completed {comparison_count} comparisons for {user_name}")

    def run_tests(self, users_base_folder: str):
        """Run tests for all user folders"""
        if not os.path.exists(users_base_folder):
            logger.error(f"Users folder not found: {users_base_folder}")
            return

        # Get all user folders (user1, user2, ..., user25)
        user_folders = []
        for item in os.listdir(users_base_folder):
            item_path = os.path.join(users_base_folder, item)
            if os.path.isdir(item_path) and item.startswith('user'):
                user_folders.append(item_path)

        user_folders.sort()

        logger.info(f"Found {len(user_folders)} user folders")

        for user_folder in user_folders:
            try:
                self.test_user_folder(user_folder)
            except Exception as e:
                logger.error(f"Error testing user folder {user_folder}: {e}")

def main():
    # Check if API is running
    tester = FaceMatchingTester()

    try:
        response = requests.get(f"{tester.api_url}/health")
        if response.status_code != 200:
            logger.error("API health check failed. Please start the API server first.")
            return
    except requests.exceptions.RequestException:
        logger.error("Cannot connect to API server. Please start the API server on localhost:8000")
        return

    # Run tests
    users_folder = os.path.join(os.path.dirname(__file__), "Users")
    tester.run_tests(users_folder)

    logger.info("All tests completed!")

if __name__ == "__main__":
    main()
