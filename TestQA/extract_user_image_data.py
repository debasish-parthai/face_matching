from pymongo import MongoClient
import time
import os
import json
import requests
import cv2
import numpy as np
import logging
import tempfile
import shutil

QA_DEST_URI = 'mongodb://normaluser:uE8WOY76HWt5ajSeeeYY@13.202.186.136:27018/matchmaking_db_prod'
QA_DEST_DB_NAME = 'matchmaking_db_prod'

# Base URL for images
BASE_IMAGE_URL = 'https://apps.parthaisolutions.com/ai-matchmaking/api/matrimony/'

# Import the ImprovedFaceMatcherInsightFace class from the other file
import sys

# Add parent directory to path to import ImprovedFaceMatcherInsightFace
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_matcher_optimize_for_script import ImprovedFaceMatcherInsightFace


def download_image(url: str, save_path: str) -> bool:
    """Download image from URL and save to specified path"""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to download {url}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False


def extract_user_image_data():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize face matcher
    face_matcher = ImprovedFaceMatcherInsightFace(model_name='buffalo_l', use_gpu=False)

    client = MongoClient(QA_DEST_URI)
    db = client[QA_DEST_DB_NAME]

    collection_user = db['user']
    collection_user_registration = db['user_registration']

    # Create Users directory
    users_dir = os.path.join(os.path.dirname(__file__), "Users_Final")
    os.makedirs(users_dir, exist_ok=True)

    # Step 1: Get user_ids from user collection where platform_live=true and gender="Male"
    user_query = {
        "platform_live": True,
        "gender": "Male"
    }

    user_ids = [user['user_id'] for user in collection_user.find(user_query, {"user_id": 1})]

    print(f"Found {len(user_ids)} male users with platform_live=true")

    if not user_ids:
        print("No users found matching criteria")
        return []

    # Step 2: Query user_registration collection for these user_ids
    # Filter for users with primary_photo and at least one additional_photo
    registration_query = {
        "user_id": {"$in": user_ids},
        "photos.primary_photo": {"$exists": True, "$ne": None},
        # "photos.additional_photos": {"$exists": True, "$ne": []},
    }

    # Project only the photos field
    projection = {
        "user_id": 1,
        "photos": 1
    }

    users_data = list(collection_user_registration.find(registration_query, projection))

    print(f"Found {len(users_data)} users with primary_photo and at least one additional_photo")

    count = 0
    # Process each user
    for user_idx, user_data in enumerate(users_data, 1):
        user_id = user_data.get('user_id')
        photos = user_data.get('photos', {})
        primary_photo = photos.get('primary_photo')
        additional_photos = photos.get('additional_photos', [])

        # Create user folder
        user_folder = os.path.join(users_dir, f"user{user_idx}")
        os.makedirs(user_folder, exist_ok=True)

        print(f"\nProcessing User {user_idx}: {user_id}")

        # Create temporary directory for downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            downloaded_images = []
            user_metadata = {
                "user_id": user_id,
                "user_number": user_idx,
                "primary_photo": None,
                "additional_photos": [],
                "cropped_images": []
            }

            # Download and process primary photo
            if primary_photo and 'url' in primary_photo:
                primary_url = primary_photo['url']
                if not primary_url.startswith('http'):
                    primary_url = BASE_IMAGE_URL + primary_url

                primary_filename = f"user{user_idx}_primary.jpg"
                primary_path = os.path.join(temp_dir, primary_filename)

                if download_image(primary_url, primary_path):
                    print(f"  Downloaded primary photo")
                    downloaded_images.append(primary_path)

                    # Extract faces from primary photo
                    faces = face_matcher.extract_faces(
                        primary_path,
                        save_cropped_faces=True,
                        output_dir=user_folder
                    )

                    if faces:
                        # Rename cropped face to standard naming
                        if 'cropped_image_path' in faces[0]:
                            original_crop_path = faces[0]['cropped_image_path']
                            new_crop_path = os.path.join(user_folder, f"user{user_idx}_primary.jpg")
                            if os.path.exists(original_crop_path):
                                shutil.move(original_crop_path, new_crop_path)
                                
                                # Save face embedding
                                embedding_filename = f"user{user_idx}_primary_embedding.npy"
                                embedding_path = os.path.join(user_folder, embedding_filename)
                                np.save(embedding_path, faces[0]['embedding'])

                                # Remove original downloaded image after successful face extraction
                                try:
                                    os.remove(primary_path)
                                except OSError as e:
                                    logging.warning(f"Could not remove original primary image {primary_path}: {e}")
                                user_metadata["primary_photo"] = {
                                    "original_url": primary_url,
                                    "cropped_path": f"user{user_idx}_primary.jpg",
                                    "embedding_path": embedding_filename,
                                    "face_detected": True,
                                    "face_info": {
                                        "bbox": faces[0]['bbox'],
                                        "det_score": faces[0]['det_score'],
                                        "age": faces[0].get('age'),
                                        "gender": faces[0].get('gender')
                                    }
                                }
                                user_metadata["cropped_images"].append(f"user{user_idx}_primary.jpg")
                    else:
                        # Save original if no face detected
                        shutil.copy2(primary_path, os.path.join(user_folder, f"user{user_idx}_primary.jpg"))
                        user_metadata["primary_photo"] = {
                            "original_url": primary_url,
                            "cropped_path": f"user{user_idx}_primary.jpg",
                            "face_detected": False
                        }
                        user_metadata["cropped_images"].append(f"user{user_idx}_primary.jpg")

            # Download and process additional photos
            for photo_idx, photo in enumerate(additional_photos):
                if 'url' in photo:
                    photo_url = photo['url']
                    if not photo_url.startswith('http'):
                        photo_url = BASE_IMAGE_URL + photo_url

                    additional_filename = f"user{user_idx}_additional_{photo_idx + 1}.jpg"
                    additional_path = os.path.join(temp_dir, additional_filename)

                    if download_image(photo_url, additional_path):
                        print(f"  Downloaded additional photo {photo_idx + 1}")
                        downloaded_images.append(additional_path)

                        # Extract faces from additional photo
                        faces = face_matcher.extract_faces(
                            additional_path,
                            save_cropped_faces=True,
                            output_dir=user_folder
                        )

                        if faces:
                            # Rename cropped face to standard naming
                            if 'cropped_image_path' in faces[0]:
                                original_crop_path = faces[0]['cropped_image_path']
                                new_crop_path = os.path.join(user_folder, f"user{user_idx}_additional_{photo_idx + 1}.jpg")
                                if os.path.exists(original_crop_path):
                                    shutil.move(original_crop_path, new_crop_path)
                                    
                                    # Save face embedding
                                    embedding_filename = f"user{user_idx}_additional_{photo_idx + 1}_embedding.npy"
                                    embedding_path = os.path.join(user_folder, embedding_filename)
                                    np.save(embedding_path, faces[0]['embedding'])

                                    # Remove original downloaded image after successful face extraction
                                    try:
                                        os.remove(additional_path)
                                    except OSError as e:
                                        logging.warning(f"Could not remove original additional image {additional_path}: {e}")
                                    user_metadata["additional_photos"].append({
                                        "original_url": photo_url,
                                        "cropped_path": f"user{user_idx}_additional_{photo_idx + 1}.jpg",
                                        "embedding_path": embedding_filename,
                                        "face_detected": True,
                                        "face_info": {
                                            "bbox": faces[0]['bbox'],
                                            "det_score": faces[0]['det_score'],
                                            "age": faces[0].get('age'),
                                            "gender": faces[0].get('gender')
                                        }
                                    })
                                    user_metadata["cropped_images"].append(f"user{user_idx}_additional_{photo_idx + 1}.jpg")
                        else:
                            # Save original if no face detected
                            shutil.copy2(additional_path, os.path.join(user_folder, f"user{user_idx}_additional_{photo_idx + 1}.jpg"))
                            user_metadata["additional_photos"].append({
                                "original_url": photo_url,
                                "cropped_path": f"user{user_idx}_additional_{photo_idx + 1}.jpg",
                                "face_detected": False
                            })
                            user_metadata["cropped_images"].append(f"user{user_idx}_additional_{photo_idx + 1}.jpg")

        # Save metadata JSON
        metadata_path = os.path.join(user_folder, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(user_metadata, f, indent=2, default=str)

        print(f"  Saved {len(user_metadata['cropped_images'])} cropped images to {user_folder}")

        count += 1

        # Clear cache periodically to free memory
        if count % 10 == 0:
            face_matcher.clear_cache()

    print(f"\nTotal users processed: {count}")
    print(f"All cropped images saved to: {users_dir}")

    # Final cleanup
    face_matcher.clear_cache()

if __name__ == '__main__':
    extract_user_image_data()