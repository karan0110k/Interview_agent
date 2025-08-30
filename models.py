from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId
import os

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['interview_pro_ai']

class User(UserMixin):
    """User model for authentication and profile management"""
    
    collection = db['users']
    
    def __init__(self, id=None, name=None, email=None, password=None, created_at=None):
        self.id = id
        self.name = name
        self.email = email
        self.password_hash = password
        self.created_at = created_at or datetime.utcnow()
    
    def set_password(self, password):
        """Set the password hash for the user"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if the provided password matches the stored hash"""
        return check_password_hash(self.password_hash, password)
    
    def save(self):
        """Save the user to the database"""
        user_data = {
            'name': self.name,
            'email': self.email,
            'password': self.password_hash,
            'created_at': self.created_at
        }
        
        if self.id:
            self.collection.update_one({'_id': self.id}, {'$set': user_data})
        else:
            result = self.collection.insert_one(user_data)
            self.id = result.inserted_id
        
        return self
    
    @classmethod
    def get_by_id(cls, user_id):
        """Get a user by ID"""
        try:
            # Convert string ID to ObjectId if it's a string
            if isinstance(user_id, str):
                user_id = ObjectId(user_id)
            
            user_data = cls.collection.find_one({'_id': user_id})
            if user_data:
                return cls(
                    id=user_data['_id'],
                    name=user_data['name'],
                    email=user_data['email'],
                    password=user_data['password'],
                    created_at=user_data['created_at']
                )
            return None
        except Exception as e:
            return None
    
    @classmethod
    def get_by_email(cls, email):
        """Get a user by email"""
        user_data = cls.collection.find_one({'email': email})
        if user_data:
            return cls(
                id=user_data['_id'],
                name=user_data['name'],
                email=user_data['email'],
                password=user_data['password'],
                created_at=user_data['created_at']
            )
        return None

class InterviewSession:
    """Model for storing interview sessions"""
    
    collection = db['interview_sessions']
    
    def __init__(self, id=None, user_id=None, resume_text=None, resume_summary=None, 
                 difficulty=None, history=None, performance_score=None, performance_metrics=None, summary=None, created_at=None):
        self.id = id
        self.user_id = user_id
        self.resume_text = resume_text
        self.resume_summary = resume_summary
        self.difficulty = difficulty
        self.history = history or []
        self.performance_score = performance_score or 0.5
        self.performance_metrics = performance_metrics or {}
        self.summary = summary or {}
        self.created_at = created_at or datetime.utcnow()
    
    def save(self):
        """Save the interview session to the database"""
        session_data = {
            'user_id': self.user_id,
            'resume_text': self.resume_text,
            'resume_summary': self.resume_summary,
            'difficulty': self.difficulty,
            'history': self.history,
            'performance_score': self.performance_score,
            'performance_metrics': self.performance_metrics,
            'summary': self.summary,
            'created_at': self.created_at
        }
        
        if self.id:
            self.collection.update_one({'_id': self.id}, {'$set': session_data})
        else:
            result = self.collection.insert_one(session_data)
            self.id = result.inserted_id
        
        return self
    
    @classmethod
    def get_by_id(cls, session_id):
        """Get an interview session by ID"""
        session_data = cls.collection.find_one({'_id': session_id})
        if session_data:
            return cls(
                id=session_data['_id'],
                user_id=session_data['user_id'],
                resume_text=session_data['resume_text'],
                resume_summary=session_data['resume_summary'],
                difficulty=session_data['difficulty'],
                history=session_data['history'],
                performance_score=session_data['performance_score'],
                created_at=session_data['created_at']
            )
        return None
    
    @classmethod
    def get_by_user_id(cls, user_id):
        """Get all interview sessions for a user"""
        sessions = []
        for session_data in cls.collection.find({'user_id': user_id}).sort('created_at', -1):
            sessions.append(cls(
                id=session_data['_id'],
                user_id=session_data['user_id'],
                resume_text=session_data['resume_text'],
                resume_summary=session_data['resume_summary'],
                difficulty=session_data['difficulty'],
                history=session_data['history'],
                performance_score=session_data['performance_score'],
                created_at=session_data['created_at']
            ))
        return sessions