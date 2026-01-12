import os
import requests
import json
from pathlib import Path
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FalseNegativeTester:
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

    def remove_unwanted_fields(self, obj):
        """Recursively remove cropped_image_path and cropped_image_base64 fields from the response"""
        if isinstance(obj, dict):
            # Remove the unwanted fields
            obj.pop('cropped_image_path', None)
            obj.pop('cropped_image_base64', None)
            # Recursively process nested dictionaries and lists
            for key, value in obj.items():
                obj[key] = self.remove_unwanted_fields(value)
        elif isinstance(obj, list):
            # Process each item in the list
            for i, item in enumerate(obj):
                obj[i] = self.remove_unwanted_fields(item)
        return obj

    def save_comparison_result(self, result: dict, output_path: str):
        """Save comparison result to JSON file"""
        try:
            # Remove unwanted fields before saving
            cleaned_result = self.remove_unwanted_fields(result.copy())
            with open(output_path, 'w') as f:
                json.dump(cleaned_result, f, indent=2)
            logger.info(f"Saved result to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving result to {output_path}: {e}")

    def test_false_negative_for_user(self, current_user_folder: str, all_user_folders: List[str]):
        """Test false negatives for a specific user against all subsequent users"""
        current_user_name = os.path.basename(current_user_folder)
        logger.info(f"Testing false negatives for user: {current_user_name}")

        # Get current user's images
        current_user_images = self.get_image_files(current_user_folder)
        if not current_user_images:
            logger.warning(f"User {current_user_name} has no images, skipping")
            return

        logger.info(f"Found {len(current_user_images)} images for {current_user_name}")

        # Create false_negative_compare_results folder
        results_folder = os.path.join(current_user_folder, "false_negative_compare_results")
        os.makedirs(results_folder, exist_ok=True)

        comparison_count = 0

        # Find the index of current user in all_user_folders
        try:
            current_index = all_user_folders.index(current_user_folder)
        except ValueError:
            logger.error(f"Could not find {current_user_folder} in user folders list")
            return

        # Collect all reference images from subsequent users up to user5
        all_reference_images = []
        reference_user_mapping = []  # To track which user each reference image belongs to

        # Find the index of user5 to limit comparisons
        user5_index = None
        for i, folder in enumerate(all_user_folders):
            if os.path.basename(folder) == 'user5':
                user5_index = i
                break

        # Determine the end index for comparisons (up to user5)
        if user5_index is not None:
            end_index = min(len(all_user_folders), user5_index + 1)
        else:
            # If user5 not found, use all subsequent users (fallback)
            end_index = len(all_user_folders)

        for other_user_folder in all_user_folders[current_index + 1:end_index]:
            other_user_name = os.path.basename(other_user_folder)
            logger.info(f"Collecting images from {other_user_name}")

            # Get other user's images
            other_user_images = self.get_image_files(other_user_folder)
            if not other_user_images:
                logger.warning(f"User {other_user_name} has no images, skipping")
                continue

            all_reference_images.extend(other_user_images)
            # Track which user each image belongs to
            reference_user_mapping.extend([other_user_name] * len(other_user_images))

        if not all_reference_images:
            logger.warning(f"No reference images found for {current_user_name}, skipping")
            return

        logger.info(f"Found {len(all_reference_images)} total reference images from subsequent users (up to user5)")

        # Compare each image from current user with ALL reference images from subsequent users in batches of 5
        for i, candidate_path in enumerate(current_user_images):
            candidate_filename = os.path.basename(candidate_path)
            logger.info(f"Comparing {candidate_filename} with {len(all_reference_images)} reference images from subsequent users (up to user5)")

            # Process reference images in batches of 5
            consolidated_results = {
                'candidate_image_name': candidate_filename,
                'reference_results': []
            }

            batch_size = 5
            for batch_start in range(0, len(all_reference_images), batch_size):
                batch_end = min(batch_start + batch_size, len(all_reference_images))
                batch_reference_paths = all_reference_images[batch_start:batch_end]
                batch_user_mapping = reference_user_mapping[batch_start:batch_end]

                logger.info(f"Processing batch {batch_start//batch_size + 1}: images {batch_start+1}-{batch_end}")

                # Call API - compare current user's image with current batch
                result = self.call_match_faces_api(candidate_path, batch_reference_paths)

                if result:
                    # Add user information to each reference result and consolidate
                    for j, ref_result in enumerate(result['reference_results']):
                        if j < len(batch_user_mapping):
                            ref_result['compared_user'] = batch_user_mapping[j]
                            consolidated_results['reference_results'].append(ref_result)
                    comparison_count += 1
                else:
                    logger.error(f"Failed to get result for {candidate_filename} batch {batch_start//batch_size + 1}")

            # Save consolidated result - single file per candidate image
            result_filename = f"false_negative_{i+1:02d}_{candidate_filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('.bmp', '')}.json"
            result_path = os.path.join(results_folder, result_filename)
            self.save_comparison_result(consolidated_results, result_path)
            logger.info(f"Saved consolidated result for {candidate_filename} in single file")

        logger.info(f"Completed {comparison_count} false negative comparisons for {current_user_name}")

    def run_tests(self, users_base_folder: str):
        """Run false negative tests for all user folders"""
        if not os.path.exists(users_base_folder):
            logger.error(f"Users folder not found: {users_base_folder}")
            return

        # Get only user1, user2, user3, user4, and user5 folders
        user_folders = []
        target_users = ['user1', 'user2', 'user3', 'user4', 'user5']
        for item in os.listdir(users_base_folder):
            item_path = os.path.join(users_base_folder, item)
            if os.path.isdir(item_path) and item in target_users:
                user_folders.append(item_path)

        user_folders.sort()

        logger.info(f"Found {len(user_folders)} user folders")

        # Test false negatives for each user against subsequent users
        for user_folder in user_folders:
            try:
                self.test_false_negative_for_user(user_folder, user_folders)
            except Exception as e:
                logger.error(f"Error testing false negatives for user folder {user_folder}: {e}")

def main():
    # Check if API is running
    tester = FalseNegativeTester()

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

    logger.info("All false negative tests completed!")

if __name__ == "__main__":
    main()
