from pymongo import MongoClient
import time

QA_DEST_URI = 'mongodb://normaluser:uE8WOY76HWt5ajSeeeYY@3.7.193.35:27018/matchmaking_db_qa'
QA_DEST_DB_NAME = 'matchmaking_db_qa'


def extract_user_image_data():
    client = MongoClient(QA_DEST_URI)
    db = client[QA_DEST_DB_NAME]

    collection_user = db['user']
    collection_user_registration = db['user_registration']

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
        "photos.additional_photos": {"$exists": True, "$ne": []},
    }

    # Project only the photos field
    projection = {
        "user_id": 1,
        "photos": 1
    }

    users_data = list(collection_user_registration.find(registration_query, projection).limit(100))

    print(f"Found {len(users_data)} users with primary_photo and at least one additional_photo")

    count = 0
    # Extract and display the data
    for user_data in users_data:
        user_id = user_data.get('user_id')
        photos = user_data.get('photos', {})

        print(f"User ID: {user_id}")
        count += 1
        
    print(f"Total users processed: {count}")

if __name__ == '__main__':
    extract_user_image_data()