import os
import json
import shutil
from pathlib import Path

def cleanup_user_folder(user_folder_path):
    """
    Clean up a user folder to keep only:
    - metadata.json
    - primary image and embedding (userX_primary.jpg, userX_primary_embedding.npy)
    - first additional image and embedding (userX_additional_1.jpg, userX_additional_1_embedding.npy)
    """
    user_folder = Path(user_folder_path)
    user_name = user_folder.name

    # Get all files in the user folder
    all_files = list(user_folder.glob("*"))

    # Files to keep
    keep_files = [
        user_folder / "metadata.json",
        user_folder / f"{user_name}_primary.jpg",
        user_folder / f"{user_name}_primary_embedding.npy",
        user_folder / f"{user_name}_additional_1.jpg",
        user_folder / f"{user_name}_additional_1_embedding.npy"
    ]

    # Files to delete
    delete_files = []
    for file_path in all_files:
        if file_path not in keep_files:
            # Only delete additional images/embeddings, keep metadata.json and primary files
            if (file_path.name.startswith(f"{user_name}_additional_") and
                not file_path.name.endswith("_1.jpg") and
                not file_path.name.endswith("_1_embedding.npy")):
                delete_files.append(file_path)

    # Delete the files
    for file_path in delete_files:
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    # Update metadata.json to reflect only the kept files
    metadata_path = user_folder / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Keep only the first additional photo
            if "additional_photos" in metadata and len(metadata["additional_photos"]) > 1:
                metadata["additional_photos"] = [metadata["additional_photos"][0]]

            # Update cropped_images list to only include kept files
            kept_images = [f"{user_name}_primary.jpg", f"{user_name}_additional_1.jpg"]
            metadata["cropped_images"] = kept_images

            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Updated metadata for {user_name}")

        except Exception as e:
            print(f"Error updating metadata for {user_name}: {e}")

def main():
    users_final_path = Path("TestQA/Users_Final")

    if not users_final_path.exists():
        print(f"Path {users_final_path} does not exist!")
        return

    # Get all user folders (user1, user2, ..., user2461)
    user_folders = []
    for item in users_final_path.iterdir():
        if item.is_dir() and item.name.startswith("user"):
            try:
                # Extract number from user folder name
                user_num = int(item.name[4:])  # Remove "user" prefix
                user_folders.append((user_num, item))
            except ValueError:
                continue

    # Sort by user number
    user_folders.sort(key=lambda x: x[0])

    print(f"Found {len(user_folders)} user folders")

    # Process each user folder
    for user_num, user_folder in user_folders:
        print(f"\nProcessing {user_folder.name}...")
        cleanup_user_folder(user_folder)

    print("\nCleanup completed!")

if __name__ == "__main__":
    main()