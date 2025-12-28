"""
Enhanced Context Management System for OBO Tutor

This module provides intelligent context tracking to maintain topic continuity
in conversations while allowing natural topic transitions.
"""

import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from models import Summary, WhatsappSummary, User, Message
import logging

logger = logging.getLogger(__name__)

class ConversationContext:
    """Represents the current conversation context"""
    
    def __init__(self):
        self.current_topic: Optional[str] = None
        self.topic_keywords: List[str] = []
        self.context_strength: float = 0.0  # How confident we are about the current topic
        self.last_updated: datetime = datetime.now(timezone.utc)
        self.topic_history: List[str] = []  # Track topic changes

class ContextManager:
    """Manages conversational context for maintaining topic continuity"""
    
    # Keywords that indicate topic continuation vs new topic
    CONTINUATION_INDICATORS = [
        "what about", "how about", "and what", "also", "additionally", 
        "furthermore", "moreover", "besides", "in addition", "tell me more",
        "explain more", "can you elaborate", "what else", "anything else",
        "continue", "more about", "further", "expand on", "about it",
        "about that", "about this", "the project", "this project", "that project",
        "the same", "it", "that", "this", "more details", "overview", 
        "explain that", "tell me about it", "about the", "of the", "for the"
    ]
    
    TOPIC_CHANGE_INDICATORS = [
        "now tell me about", "let's talk about", "what is", "can you explain", 
        "I want to know about", "tell me about", "switch to", "change topic",
        "different question", "new question", "another topic", "moving on"
    ]
    
    QUESTION_WORDS = ["what", "how", "why", "when", "where", "who", "which"]
    
    # Subject areas for robotics/programming curriculum
    CURRICULUM_TOPICS = {
        "programming": ["python", "coding", "programming", "algorithm", "variable", "function", "loop", "condition", "code"],
        "robotics": ["robot", "robotics", "sensor", "actuator", "motor", "arduino", "microcontroller", "roomba", "navigation", "autonomous", "obstacle", "avoidance"],
        "electronics": ["circuit", "voltage", "current", "resistor", "capacitor", "led", "electronics", "component"],
        "algorithms": ["sorting", "searching", "recursion", "data structure", "algorithm", "complexity"],
        "embedded": ["embedded", "microcontroller", "firmware", "hardware", "pins", "gpio"],
        "projects": ["project", "roomba", "navigation", "autonomous", "build", "create", "overview", "design"]
    }

    def __init__(self):
        self.contexts: Dict[int, ConversationContext] = {}  # user_id -> context
    
    def extract_topic_from_question(self, question: str) -> Optional[str]:
        """Extract the main topic from a user's question"""
        question_lower = question.lower().strip()
        
        # Remove common question words and articles
        cleaned = re.sub(r'\b(what|how|why|when|where|who|which|is|are|the|a|an)\b', '', question_lower)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Look for curriculum topics
        for topic_category, keywords in self.CURRICULUM_TOPICS.items():
            for keyword in keywords:
                if keyword in question_lower:
                    return topic_category
        
        # Extract potential topic from noun phrases
        words = cleaned.split()
        if len(words) > 0:
            # Find the most significant word (usually a noun)
            for word in words:
                if len(word) > 3 and word not in ["about", "from", "with", "have", "does", "can"]:
                    return word
        
        return None

    def is_continuation_question(self, question: str, current_context: Optional[ConversationContext]) -> bool:
        """Determine if this question is continuing the current topic"""
        if not current_context or not current_context.current_topic:
            return False
        
        question_lower = question.lower().strip()
        
        # Check for explicit continuation indicators
        for indicator in self.CONTINUATION_INDICATORS:
            if indicator in question_lower:
                return True
        
        # Check for topic change indicators
        for indicator in self.TOPIC_CHANGE_INDICATORS:
            if indicator in question_lower:
                return False
        
        # Special check for referential questions (the, it, that, this)
        referential_words = ["the project", "this project", "that project", "it", "that", "this", "the same"]
        if any(ref in question_lower for ref in referential_words):
            return True
        
        # Check if question contains current topic keywords
        topic_mentioned = False
        for keyword in current_context.topic_keywords:
            if keyword.lower() in question_lower:
                topic_mentioned = True
                break
        
        # If no explicit indicators, check for topic keywords and question structure
        if topic_mentioned:
            return True
        
        # Check if it's a short follow-up question (likely continuation)
        words_in_question = len(question.split())
        if words_in_question <= 8 and any(qw in question_lower for qw in self.QUESTION_WORDS):
            # If it's a short question and doesn't explicitly mention a new topic, likely continuation
            has_new_topic_words = any(topic in question_lower for topic_list in self.CURRICULUM_TOPICS.values() for topic in topic_list)
            if not has_new_topic_words:
                return True
        
        return False

    def is_topic_change_question(self, question: str) -> bool:
        """Determine if this question is starting a new topic"""
        question_lower = question.lower().strip()
        
        # Check for explicit topic change indicators
        for indicator in self.TOPIC_CHANGE_INDICATORS:
            if indicator in question_lower:
                return True
        
        # Check if it's a completely new topic question
        new_topic = self.extract_topic_from_question(question)
        return new_topic is not None

    def update_context(self, user_id: int, question: str, ai_response: str) -> ConversationContext:
        """Update conversation context based on new question and response"""
        
        # Get or create context for this user
        if user_id not in self.contexts:
            self.contexts[user_id] = ConversationContext()
        
        context = self.contexts[user_id]
        
        # Determine if this is a continuation or new topic
        is_continuation = self.is_continuation_question(question, context)
        is_new_topic = self.is_topic_change_question(question)
        
        if is_new_topic and not is_continuation:
            # Starting a new topic
            new_topic = self.extract_topic_from_question(question)
            if new_topic:
                # Save previous topic to history
                if context.current_topic:
                    context.topic_history.append(context.current_topic)
                
                context.current_topic = new_topic
                context.topic_keywords = self._extract_keywords(question, ai_response)
                context.context_strength = 1.0
                logger.info(f"New topic started for user {user_id}: {new_topic}")
        
        elif is_continuation and context.current_topic:
            # Continuing current topic
            new_keywords = self._extract_keywords(question, ai_response)
            context.topic_keywords.extend(new_keywords)
            context.topic_keywords = list(set(context.topic_keywords))  # Remove duplicates
            context.context_strength = min(1.0, context.context_strength + 0.2)
            logger.info(f"Continuing topic for user {user_id}: {context.current_topic}")
        
        else:
            # Ambiguous case - check if we can extract any topic
            potential_topic = self.extract_topic_from_question(question)
            if potential_topic and potential_topic != context.current_topic:
                # Seems like a new topic
                if context.current_topic:
                    context.topic_history.append(context.current_topic)
                context.current_topic = potential_topic
                context.topic_keywords = self._extract_keywords(question, ai_response)
                context.context_strength = 0.8
            elif context.current_topic:
                # Probably continuing but not clearly indicated
                context.context_strength = max(0.3, context.context_strength - 0.1)
        
        context.last_updated = datetime.now(timezone.utc)
        
        # Clean up old contexts (keep last 5 topics in history)
        if len(context.topic_history) > 5:
            context.topic_history = context.topic_history[-5:]
        
        return context

    def _extract_keywords(self, question: str, response: str) -> List[str]:
        """Extract relevant keywords from question and response"""
        text = f"{question} {response}".lower()
        
        # Find technical terms and important nouns
        keywords = []
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        
        for word in words:
            # Skip common words
            if word not in ["the", "and", "are", "you", "can", "how", "what", "this", "that", "with", "for"]:
                # Check if it's a curriculum-related term
                found_curriculum = False
                for topic_keywords in self.CURRICULUM_TOPICS.values():
                    if word in topic_keywords:
                        keywords.append(word)
                        found_curriculum = True
                        break
                # Add other significant terms
                if not found_curriculum and len(word) > 4:
                    keywords.append(word)
        
        return list(set(keywords))

    def get_context_enhanced_summary(self, user_id: int, question: str, ai_response: str) -> str:
        """Generate a context-aware summary"""
        context = self.contexts.get(user_id)
        
        if context and context.current_topic:
            # Include topic information in summary
            summary = f"[Topic: {context.current_topic}] User question: {question} AI answer: {ai_response}"
            
            # Add continuation indicator if it's a follow-up
            if context.context_strength > 0.5:
                summary = f"[Continuing {context.current_topic}] " + summary
        else:
            # Standard summary
            summary = f"User question: {question} AI answer: {ai_response}"
        
        return summary

    def get_contextual_chat_history(self, user_id: int, db: Session, chatbox_id: Optional[int] = None, 
                                  limit: int = 10) -> str:
        """Get chat history with enhanced context awareness"""
        
        context = self.contexts.get(user_id)
        
        # Get recent summaries
        if chatbox_id:
            # For regular chat
            summaries = db.query(Summary).filter(
                Summary.user_id == user_id,
                Summary.chatbox_id == chatbox_id
            ).order_by(Summary.created_at.desc()).limit(limit).all()
        else:
            # For WhatsApp
            summaries = db.query(WhatsappSummary).filter(
                WhatsappSummary.user_id == user_id
            ).order_by(WhatsappSummary.created_at.desc()).limit(limit).all()
        
        # Organize summaries by topic relevance
        if context and context.current_topic:
            relevant_summaries = []
            other_summaries = []
            
            for summary in summaries:
                if context.current_topic.lower() in summary.summary.lower():
                    relevant_summaries.append(summary)
                else:
                    other_summaries.append(summary)
            
            # Prioritize relevant summaries
            organized_summaries = relevant_summaries + other_summaries[:max(0, limit - len(relevant_summaries))]
        else:
            organized_summaries = summaries
        
        # Build context string
        chat_history = ""
        for summary in reversed(organized_summaries):  # Reverse to get chronological order
            chat_history += summary.summary + " | "
        
        return chat_history

    def get_enhanced_prompt_context(self, user_id: int) -> str:
        """Get additional context information for the prompt"""
        context = self.contexts.get(user_id)
        
        if not context or not context.current_topic:
            return "No specific topic context."
        
        context_info = f"""
        [Current Conversation Context]
        Main Topic: {context.current_topic}
        Topic Keywords: {', '.join(context.topic_keywords)}
        Context Strength: {context.context_strength:.2f}
        Previous Topics: {', '.join(context.topic_history[-3:]) if context.topic_history else 'None'}
        
        Instructions: The user is currently focused on '{context.current_topic}'. 
        Unless they explicitly change the topic, assume follow-up questions relate to '{context.current_topic}'.
        Provide coherent, connected responses that build on previous explanations about '{context.current_topic}'.
        """
        
        return context_info

# Global context manager instance
context_manager = ContextManager()