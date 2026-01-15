import os
import re
import json
import sys

def get_users_order():
    """Get users in the same order as the cross-user matching algorithm"""
    users_dir = r"c:\Work\FaceMatching\TestQA\Users_Final"

    # Get user folders (same as algorithm - no sorting, just os.listdir() order)
    user_folders = [f for f in os.listdir(users_dir) if os.path.isdir(os.path.join(users_dir, f))]
    user_folders.sort(key=lambda x: int(x.split('user')[-1]))

    return user_folders

def count_user1_occurrences(target_user):
    """Count how many times a specific user appears as user1_folder in results"""
    folder_path = r"c:\Work\FaceMatching\TestQA\categorized_similarity_results"
    count = 0

    # Get all JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                matches = re.findall(f'"user1_folder": "{re.escape(target_user)}"', content)
                count += len(matches)

        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    return count

def verify_algorithm_correctness():
    """Verify that the cross-user matching algorithm is working correctly"""
    print("Verifying cross-user matching algorithm correctness...")

    # Get users in algorithm order
    users = get_users_order()
    print(f"Total user folders found: {len(users)}")

    # Find users with embeddings (simulate what algorithm does)
    users_with_embeddings = []
    users_dir = r"c:\Work\FaceMatching\TestQA\Users_Final"

    for folder in users:
        user_path = os.path.join(users_dir, folder)
        metadata_path = os.path.join(user_path, "metadata.json")

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Check if user has embeddings
                has_embeddings = False

                # Check primary photo
                if metadata.get("primary_photo") and metadata["primary_photo"].get("embedding_path"):
                    emb_path = os.path.join(user_path, metadata["primary_photo"]["embedding_path"])
                    if os.path.exists(emb_path):
                        has_embeddings = True

                # Check additional photos
                for photo in metadata.get("additional_photos", []):
                    if photo.get("embedding_path"):
                        emb_path = os.path.join(user_path, photo["embedding_path"])
                        if os.path.exists(emb_path):
                            has_embeddings = True
                            break

                if has_embeddings:
                    users_with_embeddings.append(folder)

            except Exception as e:
                print(f"Error checking {folder}: {e}")
                continue

    print(f"Users with embeddings: {len(users_with_embeddings)}")

    # Test with a few sample users
    test_users = []
    if len(users_with_embeddings) >= 10:
        # Test first user, middle user, and last user
        test_users = [users_with_embeddings[0], users_with_embeddings[len(users_with_embeddings)//2], users_with_embeddings[-1]]

    print("\nTesting algorithm correctness with sample users:")
    print("-" * 60)

    for test_user in test_users:
        # Find position in users_with_embeddings
        try:
            user_index = users_with_embeddings.index(test_user)
            expected_count = len(users_with_embeddings) - user_index - 1
            actual_count = count_user1_occurrences(test_user)

            status = "✓ CORRECT" if actual_count == expected_count else f"✗ INCORRECT (expected {expected_count})"

            print(f"User: {test_user}")
            print(f"  Position: {user_index + 1}/{len(users_with_embeddings)}")
            print(f"  Expected comparisons as user1: {expected_count}")
            print(f"  Actual occurrences as user1: {actual_count}")
            print(f"  Status: {status}")
            print()

        except ValueError:
            print(f"User {test_user} not found in embeddings list")
            continue

def count_specific_user(target_user):
    """Count occurrences of a specific user as user1_folder"""
    count = count_user1_occurrences(target_user)

    # Calculate expected count by finding position among users with embeddings
    users_dir = r"c:\Work\FaceMatching\TestQA\Users_Final"
    users = get_users_order()

    users_with_embeddings = []
    for folder in users:
        user_path = os.path.join(users_dir, folder)
        metadata_path = os.path.join(user_path, "metadata.json")

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Check if user has embeddings (same logic as algorithm)
                has_embeddings = False

                # Check primary photo
                if metadata.get("primary_photo") and metadata["primary_photo"].get("embedding_path"):
                    emb_path = os.path.join(user_path, metadata["primary_photo"]["embedding_path"])
                    if os.path.exists(emb_path):
                        has_embeddings = True
                    else:
                        print(f"Primary photo embedding not found for {folder}")

                # Check additional photos
                for photo in metadata.get("additional_photos", []):
                    if photo.get("embedding_path"):
                        emb_path = os.path.join(user_path, photo["embedding_path"])
                        if os.path.exists(emb_path):
                            has_embeddings = True
                            break
                        else:
                            print(f"Additional photo embedding not found for {folder}")
                if has_embeddings:
                    users_with_embeddings.append(folder)

            except Exception as e:
                continue

    expected_count = 0
    if target_user in users_with_embeddings:
        user_index = users_with_embeddings.index(target_user)
        expected_count = len(users_with_embeddings) - user_index - 1
        print(f"User '{target_user}' appears as user1_folder in {count} comparisons")
        print(f"Position in embeddings list: {user_index} (0-based)")
        print(f"Total users with embeddings: {len(users_with_embeddings)}")
        print(f"Expected comparisons: {len(users_with_embeddings)} - {user_index} - 1 = {expected_count}")
        status = "CORRECT" if count == expected_count else f"INCORRECT (missing {expected_count - count} comparisons)"
        print(f"Status: {status}")
    else:
        print(f"User '{target_user}' not found in users with embeddings")

    print(f"Total user folders: {len(users)}")
    print(f"Users with embeddings: {len(users_with_embeddings)}")

    return count

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "verify":
            verify_algorithm_correctness()
        elif sys.argv[1] == "count":
            if len(sys.argv) > 2:
                count_specific_user(sys.argv[2])
            else:
                print("Usage: python count_user1_simple.py count <username>")
        else:
            print("Usage: python count_user1_simple.py [verify|count <username>]")
    else:
        # Default: count user2463
        count_specific_user("user2463")