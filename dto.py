from pydantic import BaseModel, EmailStr, Field, field_validator
from enum import Enum
from datetime import datetime
import re




#   Pydantic BaseModels for


# roles for messages
class Role(str, Enum):
    QUESTION = "user"
    ANSWER = "gpt"

# role for users
class UserRole(str, Enum):
    STUDENT = "Student"
    TUTOR = "Tutor"
    ADMIN = "Admin"

class UserBase(BaseModel):
    first_name: str
    last_name: str
    email: str
    password: str
    phone_number: str
    role: str = UserRole.STUDENT
    age:  int
    communication_rating: int
    leadership_rating: int
    behaviour_rating: int
    responsiveness_rating: int
    difficult_concepts: str
    understood_concepts: str
    activity_summary: str
    tone_style: str

class UserBaseAdmin(BaseModel):
    id: int
    communication_rating: int
    leadership_rating: int
    behaviour_rating: int
    responsiveness_rating: int
    difficult_concepts: str
    understood_concepts: str
    activity_summary: str
  
    
class Chatbox(BaseModel):
    chat_name: str = "to be filled"
    user_id: int 

class ChatboxRequest(BaseModel):
    chat_name: str

class ChatboxRenameRequest(BaseModel):
    new_name: str
   
    
    
class Message(BaseModel):
    message: str
    message_type: str = Role
    chatbox_id: int
    user_id: int
 
    


class UserLogin(BaseModel):
    email: str
    password: str
    
    
class TokenData(BaseModel):
    email: str
    
    
class UserValidate(BaseModel):
    email: EmailStr
    phone_number: str 
    password: str = Field(..., example="P@ssw0rd")


    @field_validator('email')
    def validate_email(cls, value):
        if not re.search(r'\w+@\w+\.\w+', value):
            raise ValueError('Invalid email address')
        return value
    
    @field_validator('phone_number')
    def validate_phone_number(cls, value):
        if not re.search(r'^\+?\d{10}$', value):
            raise ValueError('Invalid phone number')
        return value
    

    @field_validator('password')
    def validate_password(cls, value):
        if len(value) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', value):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', value):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', value):
            raise ValueError('Password must contain at least one digit')
        return value
    
    class Config:
        json_schema_extra = {
            "phone number": {
                "format 1": "+94 xx xxx xxxx",
                "format 2": "  0 xx xxx xxxx"
            }
        }
    


class ChatboxUpdateRequest(BaseModel):
    chat_name: str
    user_id: int
