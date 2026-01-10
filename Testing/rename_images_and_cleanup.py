#!/usr/bin/env python3
"""
Script to rename image files in user folders and delete compare_results folders.

This script will:
1. Traverse all user folders in Testing/Users/
2. Rename all .jpg files to format: userX_01.jpg, userX_02.jpg, etc.
3. Delete all compare_results folders
"""

import os
import shutil
import re
from pathlib import Path

def rename_images_and_cleanup():
    """
    Rename images to userX_01.jpg format and delete compare_results folders
    """
    # Get the base directory (Testing folder)
    testing_dir = Path(__file__).parent
    users_dir = testing_dir / "Users"

    if not users_dir.exists():
        print(f"Users directory not found: {users_dir}")
        return

    print(f"Processing user folders in: {users_dir}")

    # Pattern to match user folders (user1, user2, etc.)
    user_folder_pattern = re.compile(r'^user\d+$')

    # Find all user folders
    user_folders = [f for f in users_dir.iterdir()
                   if f.is_dir() and user_folder_pattern.match(f.name)]

    user_folders.sort(key=lambda x: int(re.search(r'\d+', x.name).group()))

    print(f"Found {len(user_folders)} user folders: {[f.name for f in user_folders]}")

    for user_folder in user_folders:
        print(f"\nProcessing {user_folder.name}...")

        # Find all .jpg files in the user folder
        jpg_files = list(user_folder.glob("*.jpg"))
        jpg_files.sort()  # Sort to ensure consistent ordering

        if not jpg_files:
            print(f"  No .jpg files found in {user_folder.name}")
            continue

        print(f"  Found {len(jpg_files)} image files")

        # Rename images to userX_01.jpg, userX_02.jpg format
        for i, jpg_file in enumerate(jpg_files, 1):
            new_name = f"{user_folder.name}_{i:02d}.jpg"
            new_path = user_folder / new_name

            print(f"    Renaming: {jpg_file.name} -> {new_name}")
            jpg_file.rename(new_path)

        # Delete compare_results folder if it exists
        compare_results_dir = user_folder / "compare_results"
        if compare_results_dir.exists():
            print(f"  Deleting compare_results folder")
            shutil.rmtree(compare_results_dir)
        else:
            print(f"  No compare_results folder found")

    print("\nProcessing complete!")

if __name__ == "__main__":
    rename_images_and_cleanup()
