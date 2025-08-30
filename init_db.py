from pymongo import MongoClient
from werkzeug.security import generate_password_hash
import os
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['interview_pro_ai']

# Create collections if they don't exist
if 'users' not in db.list_collection_names():
    db.create_collection('users')
    print("Created 'users' collection")

if 'interview_sessions' not in db.list_collection_names():
    db.create_collection('interview_sessions')
    print("Created 'interview_sessions' collection")

# Create default admin user if it doesn't exist
admin_user = db.users.find_one({"email": "admin@example.com"})
if not admin_user:
    db.users.insert_one({
        "name": "Admin User",
        "email": "admin@example.com",
        "password": generate_password_hash("admin123"),
        "created_at": datetime.datetime.now()
    })
    print("Created default admin user (email: admin@example.com, password: admin123)")

print("Database initialization complete!")