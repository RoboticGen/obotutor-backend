"""
Chat Name Generator for OBO Tutor

This module generates meaningful chat names based on conversation content
to make the sidebar more user-friendly and descriptive.
"""

import re
from typing import List, Optional
from datetime import datetime

class ChatNameGenerator:
    """Generates meaningful names for chat conversations"""
    
    # Common topics and their descriptive names
    TOPIC_TEMPLATES = {
        "python": "Python Programming",
        "programming": "Programming Basics",
        "robotics": "Robotics Project",
        "roomba": "Roomba Navigation",
        "arduino": "Arduino Development", 
        "electronics": "Electronics & Circuits",
        "algorithms": "Algorithms & Logic",
        "embedded": "Embedded Systems",
        "sensors": "Sensors & Hardware",
        "functions": "Functions & Methods",
        "loops": "Loops & Control Flow",
        "variables": "Variables & Data",
        "debugging": "Debugging Help",
        "project": "Project Discussion"
    }
    
    # Question type patterns for more specific naming
    QUESTION_PATTERNS = {
        r"what is|explain.*about|tell me about": "Learning about",
        r"how do|how to|how can": "How to Guide",
        r"why does|why is|why do": "Understanding Why",
        r"error|bug|problem|issue|not working": "Troubleshooting",
        r"code review|check.*code|is this correct": "Code Review",
        r"homework|assignment|exercise": "Assignment Help",
        r"project.*help|help.*project": "Project Assistance"
    }
    
    # Fallback names for different scenarios
    FALLBACK_NAMES = [
        "General Discussion",
        "Programming Help",
        "Learning Session", 
        "Study Chat",
        "Q&A Session"
    ]
    
    def __init__(self):
        self.used_names = set()  # Track used names to avoid duplicates
    
    def extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from the text"""
        text_lower = text.lower()
        found_topics = []
        
        # Look for curriculum topics
        for topic, display_name in self.TOPIC_TEMPLATES.items():
            if topic in text_lower:
                found_topics.append(display_name)
        
        # Extract technical terms (capitalized words or programming terms)
        technical_terms = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b|\b(?:def|class|if|for|while|return|import)\b', text)
        for term in technical_terms[:2]:  # Limit to first 2 technical terms
            if len(term) > 2 and term.lower() not in ['the', 'and', 'you', 'can', 'how']:
                found_topics.append(term.capitalize())
        
        return list(set(found_topics))  # Remove duplicates
    
    def detect_question_type(self, question: str) -> Optional[str]:
        """Detect the type of question being asked"""
        question_lower = question.lower()
        
        for pattern, question_type in self.QUESTION_PATTERNS.items():
            if re.search(pattern, question_lower):
                return question_type
        
        return None
    
    def generate_name_from_first_message(self, question: str, response: str = "") -> str:
        """Generate a chat name based on the first message"""
        
        # Extract key topics from question and response
        topics = self.extract_key_topics(f"{question} {response}")
        
        # Detect question type
        question_type = self.detect_question_type(question)
        
        # Generate name based on available information
        name_parts = []
        
        if question_type and topics:
            # Combine question type with topic: "How to Guide: Python"
            primary_topic = topics[0]
            name = f"{question_type}: {primary_topic}"
        elif topics:
            # Just use the primary topic: "Python Programming"
            if len(topics) == 1:
                name = topics[0]
            else:
                # Multiple topics: "Python & Arduino"
                name = " & ".join(topics[:2])
        elif question_type:
            # Just question type: "How to Guide"
            name = question_type
        else:
            # Extract key words from the question
            name = self.generate_name_from_keywords(question)
        
        # Ensure name isn't too long
        if len(name) > 30:
            name = name[:27] + "..."
        
        # Make unique if needed
        name = self.make_unique_name(name)
        
        return name
    
    def generate_name_from_keywords(self, question: str) -> str:
        """Generate name from important keywords in the question"""
        
        # Remove common question words and clean up
        cleaned = re.sub(r'\b(what|how|why|when|where|who|which|is|are|can|do|does|the|a|an|about|me|you)\b', '', question.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Get important words (longer than 3 characters)
        words = [word.capitalize() for word in cleaned.split() if len(word) > 3]
        
        if words:
            # Take first 2-3 meaningful words
            if len(words) == 1:
                return f"{words[0]} Discussion"
            elif len(words) == 2:
                return f"{words[0]} & {words[1]}"
            else:
                return f"{words[0]} {words[1]}"
        
        # Ultimate fallback
        return self.get_fallback_name()
    
    def make_unique_name(self, base_name: str) -> str:
        """Make the name unique by adding a number if needed"""
        if base_name not in self.used_names:
            self.used_names.add(base_name)
            return base_name
        
        # Add number to make unique
        counter = 2
        while f"{base_name} ({counter})" in self.used_names:
            counter += 1
        
        unique_name = f"{base_name} ({counter})"
        self.used_names.add(unique_name)
        return unique_name
    
    def get_fallback_name(self) -> str:
        """Get a fallback name when nothing else works"""
        timestamp = datetime.now().strftime("%m/%d")
        for fallback in self.FALLBACK_NAMES:
            name = f"{fallback} - {timestamp}"
            if name not in self.used_names:
                self.used_names.add(name)
                return name
        
        # Ultimate fallback with timestamp
        name = f"Chat - {datetime.now().strftime('%m/%d %H:%M')}"
        self.used_names.add(name)
        return name
    
    def update_name_with_conversation(self, current_name: str, messages: List[str]) -> str:
        """Update chat name based on full conversation context"""
        
        # If current name is already meaningful, keep it
        if any(template in current_name for template in self.TOPIC_TEMPLATES.values()):
            return current_name
        
        # If it's a generic name, try to improve it
        if current_name in ["to be filled", "New Chat", "Chat"] or current_name.startswith("Chat -"):
            # Analyze all messages to find common themes
            all_text = " ".join(messages)
            topics = self.extract_key_topics(all_text)
            
            if topics:
                new_name = topics[0]
                if len(topics) > 1:
                    new_name = f"{topics[0]} & {topics[1]}"
                return self.make_unique_name(new_name)
        
        return current_name

# Global instance
chat_name_generator = ChatNameGenerator()