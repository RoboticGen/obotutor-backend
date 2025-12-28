import os
from fastapi import FastAPI,HTTPException, Depends, status,Request , Form
from typing import Annotated, Optional
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import models
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from dto import UserBase, UserLogin, Chatbox, ChatboxRequest, ChatboxRenameRequest, Message, UserValidate, TokenData , ChatboxUpdateRequest , UserBaseAdmin # ---
from database import engine, SessionLocal
from passlib.context import CryptContext
import logging
from twilio.rest import Client
from urllib.parse import parse_qs

# Removed OpenAI and LangChain imports since ML Model service handles AI processing
# from openai import OpenAI
# from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings

from sqlalchemy.orm import Session
from sqlalchemy import asc , desc

from dotenv import load_dotenv
load_dotenv()

from twillio import send_message
from ml_model_service import ml_model_service
from chat_name_generator import chat_name_generator

# Removed LangChain imports since ML Model service handles AI processing
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI

from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
import jwt
from fastapi.middleware.cors import CORSMiddleware
import markdown2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict the allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # This allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # This allows all headers
)


# google api key (not used directly anymore but kept for compatibility)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")


# twilio api keys
TWILIO_ACCOUNT_SID=os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN=os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER=os.getenv("TWILIO_NUMBER")

account_sid = TWILIO_ACCOUNT_SID
auth_token = TWILIO_AUTH_TOKEN
client = Client(account_sid, auth_token)
twilio_number = TWILIO_NUMBER 

SECRET_KEY =os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # 30 minutes






logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


models.Base.metadata.create_all(bind=engine)


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)



# Define the OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# JWT token generation and decoding
def create_jwt_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now()  + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_jwt_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_user_id_from_token(payload: dict) -> int:
    """Extract user_id from JWT payload and convert to int"""
    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise HTTPException(status_code=401, detail="Invalid token - no user ID")
    try:
        return int(user_id_str)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token - malformed user ID")




# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Admin privilege checking functions
def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    """Get current user from JWT token"""
    payload = decode_jwt_token(token)
    user_id = get_user_id_from_token(payload)
    
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Handle both user ID (for regular users) and email (for hardcoded admin)
    if isinstance(user_id, str) and "@" in user_id:
        # This is the hardcoded admin login
        if user_id == "admin@gmail.com":
            # Return a mock admin user object for hardcoded admin
            class MockAdmin:
                def __init__(self):
                    self.id = 0
                    self.email = "admin@gmail.com"
                    self.role = "Admin"
                    self.first_name = "Hardcoded"
                    self.last_name = "Admin"
            return MockAdmin()
    else:
        # This is a regular user login
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    
    raise HTTPException(status_code=401, detail="Invalid token")


def get_admin_user(current_user = Depends(get_current_user)):
    """Check if current user is admin"""
    if hasattr(current_user, 'role') and current_user.role == "Admin":
        return current_user
    elif hasattr(current_user, 'email') and current_user.email == "admin@gmail.com":
        return current_user
    else:
        raise HTTPException(status_code=403, detail="Admin privileges required")


def is_admin_or_self(current_user = Depends(get_current_user), target_user_id: int = None):
    """Check if user is admin or accessing their own data"""
    # Admin can access anything
    if hasattr(current_user, 'role') and current_user.role == "Admin":
        return True
    elif hasattr(current_user, 'email') and current_user.email == "admin@gmail.com":
        return True
    # Regular users can only access their own data
    elif target_user_id and hasattr(current_user, 'id') and current_user.id == target_user_id:
        return True
    else:
        raise HTTPException(status_code=403, detail="Access denied")



# The following functions and variables are no longer used since ML Model service handles AI processing
# Keeping them commented for reference in case of rollback

# load gemini pro model
# def load_model(model_name):
#   if model_name=="gemini-pro":
#     llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key= GOOGLE_API_KEY)
#   else:
#     llm=ChatGoogleGenerativeAI(model="gemini-pro-vision" , google_api_key= GOOGLE_API_KEY)
#   return llm

# def load_model():
#     llm = ChatOpenAI(model="gpt-4o")
#     return llm

# text_model = load_model("gemini-pro")
# text_model =  ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0.5,
#     max_tokens=1500,
# )

# embedding_model =  OpenAIEmbeddings(model="text-embedding-3-small")

# load vector store
# vectorstore = load_vector_store(directory=vector_database_path, embedding_model=embedding_model)

#prompt template
prompt_template = """
    You are an AI tutor. Adjust your response based on the following student profile, the chat history and the context.

    Adjust the your response according to chat history and profile details.

    Answer like a converation between a student and a teacher.If student say hi, or any greeting, you should respond accordingly.
    Strickly follow the curriculum content. Dont exceed the curriculum content.
    You can use the context to provide the answer to the question. If you dont have the answer in the context, you can give I dont know.
    Dont use images in the answer and Limit the answer to 1500 maximum characters.

    If the student ask question from the chat history, you should provide the answer but dont give any answer outside the curriculum content.

    [Student profile]
    Age: {age}
    Communication Rating: {communication_rating}
    Leadership Rating: {leadership_rating}
    Behaviour Rating: {behaviour_rating}
    Responsiveness Rating: {responsiveness_rating}
    Difficult Concepts: {difficult_concepts}
    Understood Concepts: {understood_concepts}
    Activity Summary: {activity_summary}

    [Profile Tone]
    Tone Style: {tone_style}

    [chat history]
    previous chat history: {chat_history}
    
    [Context]
    Curriculum: RoboticGen Academy, Notes Content: {context},

    [student question]
    {question}

    If you have no context or outside the curriculum , Tell straightly that I dont know the answer and Dont give any references.

    If the question is related to the curriculum,you should provide at least two more sources like website links for the student to refer to this section.
    underline if you give any links.
   

    [tutor response]

    """
whatsapp_prompt_template = """
    You are an AI tutor assisting a student. Use the provided student profile, chat history, and curriculum context to tailor your response.

    Respond conversationally as a tutor would. Greet the student appropriately if they initiate with a greeting. Adhere strictly to RoboticGen Academy’s curriculum content and do not exceed it.

    Use the provided context to answer questions. If the answer is not within the context, respond with "I don't know." Limit answers to 800 characters and avoid using images.

    If the question is related to Roboticgen curriculum,Provide links, such as relevant curriculum links or YouTube videos. Do not provide content outside the curriculum.

    If the student ask question from the chat history, you should provide the answer but dont give any answer outside the curriculum content.

    If you have no context or outside the curriculum , Tell straightly that I dont know the answer and Dont give any references.

    If the question is related to the curriculum,you should provide more sources like website links , youtube video links for the student to refer to.
    underline if you give any links.

    [Student Profile]
    Age: {age}
    Communication Rating: {communication_rating}
    Leadership Rating: {leadership_rating}
    Behaviour Rating: {behaviour_rating}
    Responsiveness Rating: {responsiveness_rating}
    Difficult Concepts: {difficult_concepts}
    Understood Concepts: {understood_concepts}
    Activity Summary: {activity_summary}

    [Tone]
    Style: {tone_style}

    [Chat History]
    Previous Conversations: {chat_history}

    [Context]
    Curriculum Topics: Programming and Algorithms, Electronics, and Embedded Systems
    Notes Content: {context}

    [Student Question]
    {question}

    [Tutor Response]
    Respond based on the curriculum and context. If unable to answer based on the curriculum, state "I don't know" without any additional references.
    """




history_summarize_prompt_template = """You are an assistant tasked with summarizing text for retrieval.
Summarize the student question and tutor answer in a concise manner.It should be a brief summary of the conversation.
But include the main points of the conversation. Add the student question in the summary.

student question: {human_question}
tutor answer: {ai_answer}

Summary:
"""





# load dependency
db_dependency = Annotated[Session, Depends(get_db)]




#============convert base 64 to image==================

# import base64
# from PIL import Image

# def convert_base64_to_image(base64_string):
#     base64_string = base64_string.split(",")[1]
#     imgdata = base64.b64decode(base64_string)
#     return imgdata











# ===================Whatsapp endpoints===========================




# ask question and get answer from the chatbot in the WHATSAPP 
@app.post("/api/whatsapp")
async def reply(question: Request,db: db_dependency):
    phone_number = parse_qs(await question.body())[b'WaId'][0].decode('utf-8')
    message_body = parse_qs(await question.body())[b'Body'][0].decode('utf-8')

    print(phone_number)
    print(message_body)

    local_phone_number = "0" + phone_number[2:]  

    print(local_phone_number)  

    try:
        user = db.query(models.User).filter(models.User.phone_number == local_phone_number).first()

        print("user", user)
        
        if user is not None:
            quiries = db.query(models.WhatsappSummary).filter(models.WhatsappSummary.user_id == user.id).order_by(models.WhatsappSummary.created_at.desc()).offset(0).limit(20).all()
            print("queries", quiries)
            chat_history = ""
            for q in quiries:
                chat_history += q.summary + ","
            print(chat_history)
            
            # Call ML Model service instead of local processing
            chat_response = await ml_model_service.query_model(message_body)
            
            print("chat_response", chat_response)
            print("chat_response",type(chat_response) )
            send_message(phone_number, chat_response.get('result') , chat_response.get('relevant_images'))
            
            # Create a simple summary for WhatsApp chat history
            summary = f"User question: {message_body} AI answer: {chat_response.get('result')}"
            db_query = models.WhatsappSummary(summary=summary,user_id=user.id, phone_number=local_phone_number) 
            db.add(db_query)
            db.commit()




        else:
            chat_response = "Hello, I am a chatbot. You have not signed up yet."
            send_message(phone_number, chat_response)
    except:
        send_message(phone_number, "wait")
  
    # try:

    #     chat_response = "Hello, I am a chatbot. I am still learning. Please wait for a moment."
    #     send_message("+94722086410", chat_response)
    # except:
    #     send_message("+94722086410", "wait")






# ======================== API ENDPOINTS ========================

# OAuth2-compliant token endpoint for authentication
@app.post("/token", status_code=status.HTTP_200_OK)
async def login_for_access_token(db: db_dependency, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2-compliant token endpoint that accepts username (email) and password
    Returns access_token and token_type for both regular users and admin
    """
    
    # Check if it's the hardcoded admin
    if form_data.username == "admin@gmail.com" and form_data.password == "adminObotutor123$":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_jwt_token({"sub": form_data.username}, expires_delta=access_token_expires)
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
    
    # Check regular users
    user_db = db.query(models.User).filter(models.User.email == form_data.username).first()
    
    if user_db is None or not pwd_context.verify(form_data.password, user_db.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_jwt_token({"sub": str(user_db.id)}, expires_delta=access_token_expires)
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


# --TO DO--
# password hashing and match databases hashed password
# login
@app.post("/api/login" , status_code=status.HTTP_200_OK)
async def login_user(user: UserLogin, db: db_dependency):
    user_db = db.query(models.User).filter(models.User.email == user.email).first()
    
    if user_db is None or not pwd_context.verify(user.password, user_db.password):
        raise HTTPException(status_code=404, detail="Invalid Credentials")
    
    # Generate JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_jwt_token({"sub": str(user_db.id)}, expires_delta=access_token_expires)
    
    return {
        "user_details": user_db, 
        "access_token": access_token, 
        "token_type": "bearer"
    }


@app.post("/api/admin/login" , status_code=status.HTTP_200_OK)
async def login_user(user: UserLogin, db: db_dependency):
   
    
    if( user.email == "admin@gmail.com" and user.password == "adminObotutor123$"):
    
    # Generate JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_jwt_token({"sub": user.email}, expires_delta=access_token_expires)
        
        
        return {
            "user_details": user.email, 
            "access_token": access_token, 
            "token_type": "bearer"
        }
    else:
        raise HTTPException(status_code=404, detail="Invalid Credentials")
    





# --TO DO--
# JWT token generation


# sign up
@app.post("/api/signup", status_code=status.HTTP_200_OK)
async def signup_user(user: UserBase, db: db_dependency):
   

    # Custom validation checks
    try:
        #  Pydantic's validation to catch any issues

        valid_user = UserValidate(
            email=user.email, 
            password=user.password, 
            phone_number=user.phone_number)
        
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e.errors()[0]['msg'])
        )
    
    # Check if user already exists
    user_db = db.query(models.User).filter(models.User.email == user.email).first()
    if user_db is not None:
        raise HTTPException(status_code=411, detail="User already exists")
    
    
    user.password = hash_password(user.password)
    db_user = models.User(**user.model_dump())

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    user_id = db_user.id
    if user_id is None:
        raise HTTPException(status_code=404, detail="User not created")
    
    # Generate JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_jwt_token({"sub": user_id}, expires_delta=access_token_expires)
    
    return {
        "user_id": user_id, 
        "access_token": access_token, 
        "token_type": "bearer"
    }



#get user by user id
@app.get("/api/user", status_code=status.HTTP_200_OK)
async def get_user(db: db_dependency, token: str = Depends(oauth2_scheme)):
        
        payload = decode_jwt_token(token)
        user_id = payload.get("sub")
        db_user = db.query(models.User).filter(models.User.id == user_id).first()
        
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return db_user

#update user tone_style , communication_format , learning_rate  by user id
@app.put("/api/user", status_code=status.HTTP_200_OK)
async def update_user(user_update: UserBase, db: db_dependency, token: str = Depends(oauth2_scheme)):
        
        payload = decode_jwt_token(token)
        user_id = payload.get("sub")
        db_user = db.query(models.User).filter(models.User.id == user_id).first()
        
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        db_user.tone_style = user_update.tone_style
        db_user.communication_rating = user_update.communication_rating
        db_user.leadership_rating = user_update.leadership_rating
        db_user.behaviour_rating = user_update.behaviour_rating
        db_user.responsiveness_rating = user_update.responsiveness_rating
        db_user.difficult_concepts = user_update.difficult_concepts
        db_user.understood_concepts = user_update.understood_concepts
        db_user.activity_summary = user_update.activity_summary
        db_user.age = user_update.age
    
    
        db.commit()
        db.flush()
        db.refresh(db_user)
        
        return db_user

#update admin  user tone_style , communication_format , learning_rate  by user id
@app.put("/api/admin/user", status_code=status.HTTP_200_OK)
async def update_user(user_update: UserBaseAdmin, db: db_dependency, token: str = Depends(oauth2_scheme)):
        
        payload = decode_jwt_token(token)
        user_id = user_update.id
        db_user = db.query(models.User).filter(models.User.id == user_id).first()
        
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
       
        db_user.communication_rating = user_update.communication_rating
        db_user.leadership_rating = user_update.leadership_rating
        db_user.behaviour_rating = user_update.behaviour_rating
        db_user.responsiveness_rating = user_update.responsiveness_rating
        db_user.difficult_concepts = user_update.difficult_concepts
        db_user.understood_concepts = user_update.understood_concepts
        db_user.activity_summary = user_update.activity_summary
     
    
        db.commit()
        db.flush()
        db.refresh(db_user)
        
        return db_user



# create chatbox
@app.post("/api/chatbox", status_code=status.HTTP_200_OK)
async def create_chatbox(chatbox_request: ChatboxRequest, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = get_user_id_from_token(payload)
    print(user_id)
    
    # Generate a meaningful default name if none provided or if it's generic
    chat_name = chatbox_request.chat_name
    if not chat_name or chat_name.strip() == "" or chat_name in ["New Chat", "Chat", "to be filled"]:
        # Generate a temporary name that will be updated with first message
        chat_name = f"New Chat - {datetime.now().strftime('%m/%d')}"
    
    #create chatbox
    newChatBox = Chatbox(chat_name=chat_name, user_id=user_id)
    
    db_chatbox = models.Chatbox(**newChatBox.model_dump())  

    db.add(db_chatbox)
    db.commit()
    db.refresh(db_chatbox)
    
    chatbox_id = db_chatbox.id
    if chatbox_id is None:
        raise HTTPException(status_code=404, detail="Chatbox not created")
    return db_chatbox


# rename chatbox
@app.put("/api/chatbox/{chatbox_id}/rename", status_code=status.HTTP_200_OK)
async def rename_chatbox(chatbox_id: int, rename_request: ChatboxRenameRequest, db: db_dependency, token: str = Depends(oauth2_scheme)):
    """Rename a chatbox to a custom name"""
    
    payload = decode_jwt_token(token)
    user_id = get_user_id_from_token(payload)
    
    # Find the chatbox and verify ownership
    chatbox = db.query(models.Chatbox).filter(
        models.Chatbox.id == chatbox_id,
        models.Chatbox.user_id == user_id
    ).first()
    
    if not chatbox:
        raise HTTPException(status_code=404, detail="Chatbox not found or access denied")
    
    # Validate new name
    new_name = rename_request.new_name.strip()
    if not new_name or len(new_name) < 1:
        raise HTTPException(status_code=400, detail="Chat name cannot be empty")
    
    if len(new_name) > 50:
        new_name = new_name[:47] + "..."
    
    # Update the name
    old_name = chatbox.chat_name
    chatbox.chat_name = new_name
    db.commit()
    
    return {
        "message": "Chatbox renamed successfully",
        "old_name": old_name,
        "new_name": new_name,
        "chatbox_id": chatbox_id
    }


# ask question and get answer from the chatbot in the WHATSAPP
# create message
@app.post("/api/chatbox/message", status_code=status.HTTP_200_OK)
async def create_message(message: Message, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")

    newMessage = Message(message=message.message, message_type=message.message_type, chatbox_id=message.chatbox_id, user_id=user_id)


    db_message = models.Message(**newMessage.model_dump())  

    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    
    message_id = db_message.id
    user_id = db_message.user_id
    chatbox_id = db_message.chatbox_id

    user = db.query(models.User).filter(models.User.id == user_id).first()
    print(user)

    # Get recent messages from this chatbox for context
    recent_messages = db.query(models.Message).filter(
        models.Message.chatbox_id == chatbox_id
    ).order_by(models.Message.created_at.desc()).limit(10).all()
    
    # Build chat history from recent messages
    chat_context = ""
    for msg in reversed(recent_messages):  # Reverse to get chronological order
        # Skip the current message we just added
        if msg.id != message_id:
            msg_user = db.query(models.User).filter(models.User.id == msg.user_id).first()
            if msg_user:
                if msg.message_type == "Human":
                    chat_context += f"Student: {msg.message}\n"
                else:
                    chat_context += f"AI: {msg.message}\n"

    print("Chat context:", chat_context)

    try:
        # Call ML Model service with chat context
        chat_response = await ml_model_service.query_model(
            question=message.message,
            chat_history=chat_context,
            chatbox_id=chatbox_id
        )
        
        # Create a simple summary for chat history
        summary = f"User question: {message.message} AI answer: {chat_response.get('result')}"
        db_query = models.Summary(summary=summary, user_id=user_id, chatbox_id=chatbox_id)
        db.add(db_query)
        db.commit()
    except Exception as e:
        print("error" , e)
        chat_response = {'result': "Sorry. At this moment, I am unable to give the answer. Please Try again later", 'relevant_images':[]}
        summary = "User question: " + message.message + " AI answer: " + chat_response.get('result')
        db_query = models.Summary(summary=summary, user_id=user_id, chatbox_id=chatbox_id)
        db.add(db_query)
        db.commit()

    related_images = ''
    relevant_images_list = chat_response.get('relevant_images', [])
    if relevant_images_list:
        if isinstance(relevant_images_list, list):
            related_images = ",".join(str(img) for img in relevant_images_list)
        else:
            related_images = str(relevant_images_list)
    print(related_images)

    # add chat response to db
    db_message = models.Message(message=chat_response.get('result'), message_type="gpt", chatbox_id=chatbox_id, user_id=user_id , related_images= related_images)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)

    # Auto-update chat name if this is the first user message or has generic name
    chatbox = db.query(models.Chatbox).filter(models.Chatbox.id == chatbox_id).first()
    if chatbox:
        # Check if this is the first meaningful exchange (2-3 messages: user + bot + maybe another user)
        total_messages = db.query(models.Message).filter(models.Message.chatbox_id == chatbox_id).count()
        
        # Update name if it's generic or this is one of the first few messages
        should_update_name = (
            total_messages <= 4 and  # First few messages 
            (chatbox.chat_name.startswith("New Chat") or 
             chatbox.chat_name in ["to be filled", "Chat", "General Discussion"] or
             "Chat - " in chatbox.chat_name)
        )
        
        if should_update_name:
            # Generate meaningful name from the first user message and AI response
            new_name = chat_name_generator.generate_name_from_first_message(
                message.message, 
                chat_response.get('result', '')
            )
            
            # Update the chatbox name
            chatbox.chat_name = new_name
            db.commit()
            print(f"Updated chat name to: {new_name}")

    return JSONResponse({"message": "Message created successfully", "result": chat_response.get('result') , "relevant_images": related_images} )





# delete chatbox
@app.delete("/api/chatbox/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(chat_id: int, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    db_chatbox = db.query(models.Chatbox).filter(models.Chatbox.id == chat_id).first()
    
    if db_chatbox is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chatbox not found")

    db.delete(db_chatbox)
    db.commit()
    
    return {"detail": "Message deleted successfully"}



# delete user
@app.delete("/api/user", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message( db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")

    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    
    if db_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    db.delete(db_user)
    db.commit()
    
    return {"detail": "User deleted successfully"}


# irrelavant
# get all messages by user id
@app.get("/api/messages/{chat_id}" , status_code=status.HTTP_200_OK)
async def read_message(chat_id: int, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")
    db_messages = db.query(models.Message).filter(
        models.Message.chatbox_id == chat_id,
        models.Message.user_id == user_id
    ).all()
    if db_messages is None:
        raise HTTPException(status_code=404, detail="Messages not found")
    return db_messages



# get all chatboxes by user id
@app.get("/api/chatbox/user" , status_code=status.HTTP_200_OK)
async def get_chatboxes( db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")

    db_chatboxes = db.query(models.Chatbox).filter(models.Chatbox.user_id == user_id).order_by(models.Chatbox.created_at.desc()).all()
    if db_chatboxes is None:
        raise HTTPException(status_code=404, detail="Chatboxes not found")
    return db_chatboxes

# get all chatboxes by chat id
@app.get("/api/chatbox/{chat_id}" , status_code=status.HTTP_200_OK)
async def get_chatboxes(chat_id: int, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    db_chatboxes = db.query(models.Chatbox).filter(models.Chatbox.id == chat_id).first()
    if db_chatboxes is None:
        raise HTTPException(status_code=404, detail="Chatboxes not found")
    return db_chatboxes

#delete chatbox by chat id and user id
@app.delete("/api/chatbox/{chat_id}" , status_code=status.HTTP_200_OK)
async def delete_chatbox(chat_id: int, db: db_dependency, token: str = Depends(oauth2_scheme)):
        
        payload = decode_jwt_token(token)
        user_id = payload.get("sub")


        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token or missing user ID")



        print(user_id , chat_id)
        db_chatbox = db.query(models.Chatbox).filter(models.Chatbox.id == chat_id ,models.Chatbox.user_id == user_id ).first()
        
        if db_chatbox is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chatbox not found")
        
        db.delete(db_chatbox)
        db.commit()
        
        # return all chatboxes by user id
        all_chatboxes = db.query(models.Chatbox).filter(models.Chatbox.user_id == user_id).all()
        
        return all_chatboxes


#update chatboxname by chat id
@app.put("/api/chatbox/{chat_id}" , status_code=status.HTTP_200_OK)
async def update_chatbox(chat_id: int, chatbox_update: ChatboxUpdateRequest, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")
    db_chatbox = db.query(models.Chatbox).filter(models.Chatbox.id == chat_id).first()
    
    if db_chatbox is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chatbox not found")
    
    db_chatbox.chat_name = chatbox_update.chat_name
    db_chatbox.user_id = user_id

    db.commit()
    db.flush()
    db.refresh(db_chatbox)

    # return all chatboxes by user id
    all_chatboxes = db.query(models.Chatbox).filter(models.Chatbox.user_id == db_chatbox.user_id).order_by(models.Chatbox.created_at.desc()).all()
    
    return all_chatboxes

# Auto-generate meaningful chat name based on conversation content
@app.put("/api/chatbox/{chat_id}/generate-name" , status_code=status.HTTP_200_OK)
async def auto_generate_chat_name(chat_id: int, db: db_dependency, token: str = Depends(oauth2_scheme)):
    """
    Automatically generate a meaningful name for the chat based on conversation content
    """
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")
    
    # Get the chatbox
    db_chatbox = db.query(models.Chatbox).filter(
        models.Chatbox.id == chat_id,
        models.Chatbox.user_id == user_id
    ).first()
    
    if db_chatbox is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chatbox not found")
    
    # Get all messages from this chat
    messages = db.query(models.Message).filter(
        models.Message.chatbox_id == chat_id
    ).order_by(models.Message.created_at.asc()).all()
    
    if not messages:
        raise HTTPException(status_code=400, detail="Cannot generate name for empty chat")
    
    # Get the first user message for primary analysis
    first_user_message = next((msg.message for msg in messages if msg.message_type == "user"), None)
    first_ai_message = next((msg.message for msg in messages if msg.message_type == "gpt"), None)
    
    if not first_user_message:
        raise HTTPException(status_code=400, detail="No user messages found")
    
    # Generate name based on conversation
    if len(messages) <= 2:
        # Use first message method
        new_name = chat_name_generator.generate_name_from_first_message(
            first_user_message, 
            first_ai_message or ""
        )
    else:
        # Use full conversation analysis
        all_messages = [msg.message for msg in messages]
        new_name = chat_name_generator.update_name_with_conversation(
            db_chatbox.chat_name, 
            all_messages
        )
    
    # Update the chatbox name
    old_name = db_chatbox.chat_name
    db_chatbox.chat_name = new_name
    db.commit()
    db.refresh(db_chatbox)
    
    return {
        "message": "Chat name updated successfully",
        "old_name": old_name,
        "new_name": new_name,
        "chatbox": db_chatbox
    }

# Batch update all generic chat names for a user
@app.put("/api/user/chatboxes/generate-names", status_code=status.HTTP_200_OK)
async def batch_generate_chat_names(db: db_dependency, token: str = Depends(oauth2_scheme)):
    """
    Batch update all generic chat names for the current user
    """
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")
    
    # Get all chatboxes for this user with generic names
    generic_names = ["to be filled", "New Chat", "Chat", "General Discussion"]
    
    chatboxes = db.query(models.Chatbox).filter(
        models.Chatbox.user_id == user_id
    ).all()
    
    updated_chats = []
    skipped_chats = []
    
    for chatbox in chatboxes:
        # Skip if name is already meaningful
        is_generic = (
            chatbox.chat_name in generic_names or 
            chatbox.chat_name.startswith("New Chat -") or
            chatbox.chat_name.startswith("Chat -")
        )
        
        if not is_generic:
            skipped_chats.append({
                "id": chatbox.id,
                "name": chatbox.chat_name,
                "reason": "Already has meaningful name"
            })
            continue
        
        # Get messages for this chat
        messages = db.query(models.Message).filter(
            models.Message.chatbox_id == chatbox.id
        ).order_by(models.Message.created_at.asc()).all()
        
        if not messages:
            skipped_chats.append({
                "id": chatbox.id,
                "name": chatbox.chat_name,
                "reason": "No messages found"
            })
            continue
        
        # Generate new name
        first_user_message = next((msg.message for msg in messages if msg.message_type == "user"), None)
        first_ai_message = next((msg.message for msg in messages if msg.message_type == "gpt"), None)
        
        if not first_user_message:
            skipped_chats.append({
                "id": chatbox.id,
                "name": chatbox.chat_name,
                "reason": "No user messages found"
            })
            continue
        
        old_name = chatbox.chat_name
        new_name = chat_name_generator.generate_name_from_first_message(
            first_user_message, 
            first_ai_message or ""
        )
        
        # Update the name
        chatbox.chat_name = new_name
        updated_chats.append({
            "id": chatbox.id,
            "old_name": old_name,
            "new_name": new_name
        })
    
    # Commit all changes
    if updated_chats:
        db.commit()
    
    return {
        "message": f"Updated {len(updated_chats)} chat names",
        "updated_chats": updated_chats,
        "skipped_chats": skipped_chats,
        "summary": {
            "total_chats": len(chatboxes),
            "updated": len(updated_chats),
            "skipped": len(skipped_chats)
        }
    }
async def get_student(email: str, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")
    db_user = db.query(models.User).filter(models.User.email == email).first()
    
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

#get all students with email without token
# @app.get("/api/students" , status_code=status.HTTP_200_OK)
# async def get_students(db: db_dependency):
        
#         db_users = db.query(models.User).all()
#         if db_users is None:
#             raise HTTPException(status_code=404, detail="Users not found")
#         return db_users

#get all students with email with token
@app.get("/api/students" , status_code=status.HTTP_200_OK)
async def get_students(db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")
    db_users = db.query(models.User).all()
    if db_users is None:
        raise HTTPException(status_code=404, detail="Users not found")
    return db_users

#delete student by email
@app.delete("/api/student/{email}" , status_code=status.HTTP_200_OK)
async def delete_student(email: str, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")
    db_user = db.query(models.User).filter(models.User.email == email).first()
    
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(db_user)
    db.commit()
    
    return {"detail": "User deleted successfully"}

#delete student by id
@app.delete("/api/student/id/{id}" , status_code=status.HTTP_200_OK)
async def delete_student(id: int, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")
    db_user = db.query(models.User).filter(models.User.id == id).first()
    
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(db_user)
    db.commit()
    
    return {"detail": "User deleted successfully"}


# from fastapi.testclient import TestClient
# from your_app import app  # Import your FastAPI app instance
# from .schemas import UserBase

# client = TestClient(app)

# def call_create_user(user_data: dict):
#     response = client.post("/users/", json=user_data)
#     if response.status_code == 201:
#         return response.json()
#     else:
#         return {"error": response.text}



