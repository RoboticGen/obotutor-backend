import httpx
import os
from typing import Dict, Any, Optional
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class MLModelService:
    def __init__(self):
        self.base_url = os.getenv("ML_MODEL_URL", "http://127.0.0.1:8000")
        self.timeout = 30.0
    
    async def query_model(self, question: str, chat_history: str = "", chatbox_id: int = None) -> Dict[str, Any]:
        """
        Send a question to the ML Model service with chat context
        
        Args:
            question: The user's question
            chat_history: Previous conversation context
            chatbox_id: ID of the chatbox for context
            
        Returns:
            Dictionary containing 'result' and 'relevant_images'
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Send POST request with question and context
                response = await client.post(
                    f"{self.base_url}/model/query/",
                    json={
                        "question": question,
                        "chat_history": chat_history,
                        "chatbox_id": chatbox_id
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "result": data.get("result", "Sorry, I couldn't process your question."),
                        "relevant_images": data.get("relevant_images", [])
                    }
                else:
                    logger.error(f"ML Model service returned status {response.status_code}: {response.text}")
                    return {
                        "result": "Sorry, I'm having trouble accessing the knowledge base right now. Please try again later.",
                        "relevant_images": []
                    }
                    
        except httpx.TimeoutException:
            logger.error("Timeout while connecting to ML Model service")
            return {
                "result": "Sorry, the response is taking too long. Please try again.",
                "relevant_images": []
            }
        except httpx.ConnectError:
            logger.error("Failed to connect to ML Model service")
            return {
                "result": "Sorry, I'm currently unable to process your question. Please try again later.",
                "relevant_images": []
            }
        except Exception as e:
            logger.error(f"Unexpected error calling ML Model service: {str(e)}")
            return {
                "result": "Sorry, something went wrong while processing your question.",
                "relevant_images": []
            }
    
    async def health_check(self) -> bool:
        """
        Check if the ML Model service is available
        
        Returns:
            True if service is available, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/")
                return response.status_code == 200
        except:
            return False

# Create a singleton instance
ml_model_service = MLModelService()