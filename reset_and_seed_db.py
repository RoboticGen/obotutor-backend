"""
Simple database reset and seeding script
"""

import os
import sys
from sqlalchemy.orm import Session
from passlib.context import CryptContext

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import engine, SessionLocal
import models
from dto import UserRole

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)

def reset_and_seed_database():
    """Reset and populate database with sample data"""
    
    # Create fresh tables
    models.Base.metadata.drop_all(bind=engine)
    models.Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    try:
        # Sample Students
        students_data = [
            {
                "first_name": "Alice", "last_name": "Johnson", 
                "email": "alice@student.com", "password": hash_password("student123"),
                "phone_number": "0712345678", "age": 16, "role": "Student",
                "communication_rating": 4, "leadership_rating": 3, "behaviour_rating": 5, "responsiveness_rating": 4,
                "difficult_concepts": "Advanced algorithms", "understood_concepts": "Basic programming",
                "activity_summary": "Active in coding challenges", "tone_style": "Encouraging"
            },
            {
                "first_name": "Bob", "last_name": "Smith",
                "email": "bob@student.com", "password": hash_password("student123"),
                "phone_number": "0723456789", "age": 15, "role": "Student", 
                "communication_rating": 3, "leadership_rating": 2, "behaviour_rating": 4, "responsiveness_rating": 3,
                "difficult_concepts": "Object-oriented programming", "understood_concepts": "Variables, Basic syntax",
                "activity_summary": "Quiet learner, prefers hands-on exercises", "tone_style": "Patient"
            },
            {
                "first_name": "Carol", "last_name": "Davis",
                "email": "carol@student.com", "password": hash_password("student123"),
                "phone_number": "0734567890", "age": 17, "role": "Student",
                "communication_rating": 5, "leadership_rating": 4, "behaviour_rating": 5, "responsiveness_rating": 5,
                "difficult_concepts": "Machine learning concepts", "understood_concepts": "Programming fundamentals, Data analysis",
                "activity_summary": "Exceptional student with strong analytical skills", "tone_style": "Professional"
            }
        ]
        
        # Sample Tutors/Admins
        tutors_data = [
            {
                "first_name": "Dr. Sarah", "last_name": "Martinez",
                "email": "sarah@tutor.com", "password": hash_password("tutor123"),
                "phone_number": "0767890123", "age": 35, "role": "Admin",
                "communication_rating": 5, "leadership_rating": 5, "behaviour_rating": 5, "responsiveness_rating": 5,
                "difficult_concepts": "", "understood_concepts": "Computer Science, AI, Machine Learning",
                "activity_summary": "Senior tutor specializing in advanced programming", "tone_style": "Professional"
            },
            {
                "first_name": "Prof. Michael", "last_name": "Chen", 
                "email": "michael@tutor.com", "password": hash_password("tutor123"),
                "phone_number": "0778901234", "age": 42, "role": "Admin",
                "communication_rating": 5, "leadership_rating": 5, "behaviour_rating": 5, "responsiveness_rating": 4,
                "difficult_concepts": "", "understood_concepts": "Robotics, Electronics, Embedded Systems",
                "activity_summary": "Robotics expert with 15 years teaching experience", "tone_style": "Encouraging"
            }
        ]
        
        # Create students
        print("Creating students...")
        for data in students_data:
            user = models.User(**data)
            db.add(user)
            print(f"  ✓ {data['first_name']} {data['last_name']} ({data['email']})")
        
        # Create tutors
        print("\nCreating tutors...")
        for data in tutors_data:
            user = models.User(**data)
            db.add(user)
            print(f"  ✓ {data['first_name']} {data['last_name']} ({data['email']})")
        
        db.commit()
        
        # Create sample chatboxes
        print("\nCreating sample chatboxes...")
        students = db.query(models.User).filter(models.User.role == "Student").all()
        
        chatbox_data = [
            {"chat_name": "Python Basics Help", "user_id": students[0].id},
            {"chat_name": "Homework Questions", "user_id": students[0].id},
            {"chat_name": "Algorithm Practice", "user_id": students[1].id},
            {"chat_name": "Project Discussion", "user_id": students[2].id},
        ]
        
        for data in chatbox_data:
            chatbox = models.Chatbox(**data)
            db.add(chatbox)
            print(f"  ✓ {data['chat_name']}")
        
        db.commit()
        
        print("\n" + "="*60)
        print("✅ Database reset and seeded successfully!")
        print(f"Created {len(students_data)} students, {len(tutors_data)} tutors, and {len(chatbox_data)} chatboxes")
        
        print("\n📋 Login Credentials:")
        print("\n👨‍🎓 Students (Password: student123):")
        for data in students_data:
            print(f"  • {data['email']}")
        
        print("\n👩‍🏫 Tutors/Admins (Password: tutor123):")
        for data in tutors_data:
            print(f"  • {data['email']}")
            
        print("\n💡 Hardcoded Admin:")
        print("  • admin@gmail.com (Password: adminObotutor123$)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("🔄 Resetting and seeding OBO Tutor Database...")
    print("="*60)
    reset_and_seed_database()
    print("\n🎉 Ready to use! Start the server with: uvicorn main:app --reload")