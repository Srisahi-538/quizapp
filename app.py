from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, render_template_string
import json
import random
import os
import sqlite3
from datetime import datetime
from functools import wraps
import hashlib
from groq import Groq
import time
import re
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)
app.secret_key = 'quiz_master_pro_secret_key_2024'
app.config['DATABASE'] = 'quiz_database.db'

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def normalize_answer(answer):
    """
    Normalize answer for comparison - handle different variations
    """
    if not answer:
        return ""
    
    # Convert to lowercase and strip whitespace
    normalized = answer.lower().strip()
    
    # Remove extra spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove common punctuation but keep important characters
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    # Handle common variations
    variations = {
        "eulers": "euler",
        "einsteins": "einstein", 
        "russells": "russell",
        "descartes": "descartes",
        "photosynthesis": "photosynthesis",
        "bigbang": "big bang",
        "george orwell": "orwell",
        "albert einstein": "einstein",
        "gold": "au",
        "france capital": "paris",
        "hundred": "100",
        "one hundred": "100",
        "largest planet": "jupiter",
        "shakespeare tragedy": "hamlet",
        "shakespeare play": "hamlet",
        "plant food process": "photosynthesis",
        "plants make food": "photosynthesis",
        "big bang theory": "big bang",
        "universe origin": "big bang",
        "father modern physics": "einstein",
        "1984 author": "orwell",
        "eulers number": "e",
        "e number": "e",
        "2.71828": "e",
        "russells paradox": "russell",
        "set paradox": "russell",
        "rene descartes": "descartes",
        "cogito ergo sum": "descartes"
    }
    
    # Check if the normalized answer matches any variation
    if normalized in variations:
        normalized = variations[normalized]
    
    return normalized.strip()

def is_answer_correct(user_answer, correct_answer):
    """
    Check if user's answer matches the correct answer with flexible matching
    """
    if not user_answer or not correct_answer:
        return False
    
    user_norm = normalize_answer(user_answer)
    correct_norm = normalize_answer(correct_answer)
    
    # Debug info
    print(f"Comparing: User='{user_norm}' vs Correct='{correct_norm}'")
    
    # Exact match
    if user_norm == correct_norm:
        print("✅ Exact match")
        return True
    
    # Check if user answer contains the correct answer (for longer answers)
    if correct_norm in user_norm and len(correct_norm) > 2:
        print(f"✅ User answer contains correct answer: '{correct_norm}' in '{user_norm}'")
        return True
    
    # Check if correct answer contains user answer (for partial answers)
    if user_norm in correct_norm and len(user_norm) > 2:
        print(f"✅ Correct answer contains user answer: '{user_norm}' in '{correct_norm}'")
        return True
    
    # Handle number variations (e.g., "100" vs "one hundred")
    number_variations = {
        "100": ["hundred", "one hundred"],
        "1": ["one"],
        "2": ["two"],
        "3": ["three"],
        "4": ["four"],
        "5": ["five"],
        "6": ["six"],
        "7": ["seven"],
        "8": ["eight"],
        "9": ["nine"],
        "10": ["ten"]
    }
    
    for num, variations in number_variations.items():
        if (user_norm == num and correct_norm in variations) or (correct_norm == num and user_norm in variations):
            print(f"✅ Number variation match: '{user_norm}' == '{correct_norm}'")
            return True
    
    # Handle common abbreviations and variations
    common_variations = {
        "au": ["gold", "chemical symbol gold"],
        "paris": ["france capital", "capital france"],
        "jupiter": ["largest planet"],
        "hamlet": ["shakespeare tragedy", "shakespeare play"],
        "photosynthesis": ["plant food process", "plants make food"],
        "bigbang": ["big bang theory", "universe origin", "big bang"],
        "einstein": ["albert einstein", "father modern physics"],
        "orwell": ["george orwell", "1984 author"],
        "euler": ["eulers number", "e number", "2.71828"],
        "russell": ["russells paradox", "set paradox"],
        "descartes": ["rene descartes", "cogito ergo sum"]
    }
    
    # Check if either answer maps to the same variation
    for standard, variations in common_variations.items():
        user_matches = user_norm == standard or user_norm in variations
        correct_matches = correct_norm == standard or correct_norm in variations
        
        if user_matches and correct_matches:
            print(f"✅ Common variation match: '{user_norm}' -> '{standard}' <- '{correct_norm}'")
            return True
    
    # Check for word overlap (if more than 50% of words match)
    user_words = set(user_norm.split())
    correct_words = set(correct_norm.split())
    
    if user_words and correct_words:
        overlap = user_words.intersection(correct_words)
        min_len = min(len(user_words), len(correct_words))
        if min_len > 0 and len(overlap) / min_len >= 0.5:
            print(f"✅ Word overlap match: {overlap}")
            return True
    
    print(f"❌ No match found: '{user_norm}' vs '{correct_norm}'")
    return False

# Make helper functions available in templates
@app.context_processor
def utility_processor():
    return dict(is_answer_correct=is_answer_correct, normalize_answer=normalize_answer)

def setup_environment():
    """Check if Groq API key is available"""
    print("Checking Groq API configuration...")
    
    # Try to load .env file if it exists
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()
        print("Loaded .env file")
    
    groq_key = os.getenv('GROQ_API_KEY', '').strip()
    
    if groq_key and groq_key.startswith('gsk_'):
        print(f"Groq API Key: VALID (length: {len(groq_key)})")
        return True, groq_key
    elif groq_key:
        print(f"Groq API Key: Found but may be invalid (length: {len(groq_key)}, starts with: {groq_key[:10]})")
        return True, groq_key
    else:
        print("Groq API Key: NOT FOUND in environment")
        print("Make sure GROQ_API_KEY is set in your environment variables")
        return False, None

# Check API availability
API_AVAILABLE, API_KEY = setup_environment()

class QuestionVectorDB:
    """Vector database for storing and retrieving unique questions"""
    
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.collection_name = "quiz_questions"
        
        # Initialize Chroma client
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            print("ChromaDB client initialized successfully")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            # Fallback to in-memory client
            self.client = chromadb.Client()
        
        # Initialize sentence transformer model for embeddings
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Sentence transformer model loaded successfully")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.embedding_model = None
        
        # Create or get collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Storage for unique quiz questions"}
            )
            print(f"Collection '{self.collection_name}' ready")
        except Exception as e:
            print(f"Error creating collection: {e}")
            self.collection = None
    
    def _generate_question_id(self, question_text, difficulty):
        """Generate unique ID for question"""
        content = f"{question_text}_{difficulty}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_embedding(self, text):
        """Generate embedding for text"""
        if self.embedding_model is None:
            # Fallback: use simple hash-based embedding
            return [hash(text) % 10000 / 10000.0] * 384  # Mock 384-dim embedding
        
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def is_question_similar(self, question_text, difficulty, similarity_threshold=0.7):
        """Check if a similar question already exists in the database"""
        if self.collection is None:
            return False
        
        try:
            # Generate embedding for the new question
            query_embedding = self._get_embedding(question_text)
            
            # Search for similar questions
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=10,
                where={"difficulty": difficulty}
            )
            
            if results['distances'] and len(results['distances'][0]) > 0:
                # Check if any similar question exceeds threshold
                min_distance = min(results['distances'][0])
                similarity = 1 - min_distance  # Convert distance to similarity
                
                if similarity > similarity_threshold:
                    print(f"❌ Similar question found with similarity: {similarity:.3f}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking question similarity: {e}")
            return False
    
    def add_question(self, question_data):
        """Add a new question to the vector database"""
        if self.collection is None:
            return False
        
        try:
            question_text = question_data['question']
            difficulty = question_data['difficulty']
            
            # Generate unique ID
            question_id = self._generate_question_id(question_text, difficulty)
            
            # Generate embedding
            embedding = self._get_embedding(question_text)
            
            # Add to collection
            self.collection.add(
                ids=[question_id],
                embeddings=[embedding],
                documents=[question_text],
                metadatas=[{
                    "difficulty": difficulty,
                    "category": question_data.get('category', 'English Language'),
                    "answer": question_data.get('answer', ''),
                    "explanation": question_data.get('explanation', ''),
                    "created_at": datetime.now().isoformat(),
                    "is_llm_generated": question_data.get('is_llm_generated', True)
                }]
            )
            
            print(f"✅ Question added to vector DB: {question_text[:80]}...")
            return True
            
        except Exception as e:
            print(f"Error adding question to vector DB: {e}")
            return False
    
    def get_question_count(self, difficulty=None, category=None):
        """Get total number of questions in database"""
        if self.collection is None:
            return 0
        
        try:
            where_conditions = {}
            if difficulty:
                where_conditions["difficulty"] = difficulty
            if category:
                where_conditions["category"] = category
                
            if where_conditions:
                result = self.collection.get(where=where_conditions)
            else:
                result = self.collection.get()
                
            return len(result['ids'])
        except Exception as e:
            print(f"Error getting question count: {e}")
            return 0

    def reset_database(self):
        """Reset the entire vector database"""
        try:
            # Delete the persistence directory
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                print(f"🗑️ Vector database reset: {self.persist_directory}")
            
            # Reinitialize
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Storage for unique quiz questions"}
            )
            print("🔄 Vector database reinitialized")
            return True
        except Exception as e:
            print(f"❌ Error resetting database: {e}")
            return False

    def get_all_questions(self):
        """Get all questions from the vector database for debugging"""
        if self.collection is None:
            return []
        
        try:
            result = self.collection.get()
            questions = []
            for i in range(len(result['ids'])):
                questions.append({
                    'id': result['ids'][i],
                    'question': result['documents'][i],
                    'metadata': result['metadatas'][i]
                })
            return questions
        except Exception as e:
            print(f"Error getting all questions: {e}")
            return []

class DynamicQuestionGenerator:
    def __init__(self, api_available, api_key):
        self.api_available = api_available
        self.api_key = api_key
        self.vector_db = QuestionVectorDB()
        self.current_quiz_questions = set()  # Track questions in current quiz only
        self.current_quiz_embeddings = []  # Track embeddings for current quiz
        
        status = "ACTIVE" if api_available else "INACTIVE"
        if api_available and api_key:
            key_preview = api_key[:10] + "..." if len(api_key) > 10 else api_key
            print(f"Dynamic Question Generator - API: {status} | Key: {key_preview}")
        else:
            print(f"Dynamic Question Generator - API: {status} | No key available")
        
        if api_available and api_key:
            try:
                self.client = Groq(api_key=api_key)
                print("Groq client successfully initialized")
            except Exception as e:
                print(f"WARNING: Failed to initialize Groq client: {e}")
                self.api_available = False
                self.client = None
        else:
            self.client = None

        # Print vector database stats
        total_questions = self.vector_db.get_question_count()
        print(f"Vector Database: {total_questions} total questions stored")
        for difficulty in ['easy', 'medium', 'hard', 'very hard']:
            count = self.vector_db.get_question_count(difficulty)
            print(f"  - {difficulty}: {count} questions")

    def _get_llm_client(self):
        """Get Groq client - use cached client instance"""
        if not self.api_available or not self.api_key:
            raise Exception("Groq API not available. Please check your GROQ_API_KEY environment variable.")
        
        if self.client is None:
            self.client = Groq(api_key=self.api_key)
        
        return self.client

    def _clean_json_response(self, content):
        """Clean and extract JSON from LLM response"""
        # Remove markdown code blocks if present
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        # Find JSON object
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")
            
        json_str = content[start_idx:end_idx]
        
        # Try to parse JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            json_str = re.sub(r"'", '"', json_str)  # Replace single quotes with double quotes
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format: {str(e)}")

    def _get_temperature(self, difficulty, attempt):
        """Get appropriate temperature based on difficulty level"""
        base_temperatures = {
            'easy': 0.7,      # Lower temperature for faster, more consistent responses
            'medium': 0.8,    # Moderate creativity
            'hard': 0.9,      # Higher creativity
            'very hard': 1.0  # Maximum creativity but within reasonable limits
        }
        
        base_temp = base_temperatures.get(difficulty, 0.8)
        # Slight increase temperature for retry attempts
        temperature = base_temp + (attempt * 0.1)
        return min(temperature, 1.2)  # Lower cap

    def _validate_difficulty(self, question, intended_difficulty):
        """More flexible difficulty validation"""
        # Be very permissive with difficulty validation to allow more unique questions
        return True

    def _is_question_similar_to_current_quiz(self, question_text, similarity_threshold=0.6):
        """Check if question is similar to any question in the current quiz"""
        if not self.current_quiz_embeddings:
            return False
        
        try:
            # Generate embedding for the new question
            new_embedding = self.vector_db._get_embedding(question_text)
            
            # Check similarity with all questions in current quiz
            for existing_embedding in self.current_quiz_embeddings:
                # Calculate cosine similarity
                if len(new_embedding) != len(existing_embedding):
                    continue
                
                # Convert to numpy arrays for efficient calculation
                new_emb = np.array(new_embedding)
                existing_emb = np.array(existing_embedding)
                
                # Calculate cosine similarity
                similarity = np.dot(new_emb, existing_emb) / (np.linalg.norm(new_emb) * np.linalg.norm(existing_emb))
                
                if similarity > similarity_threshold:
                    print(f"❌ Question too similar to current quiz question (similarity: {similarity:.3f})")
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking current quiz similarity: {e}")
            return False

    def _build_optimized_prompt(self, difficulty, category, attempt_number):
        """Improved prompt for better question generation"""
        base_prompt = f"""Create a unique {difficulty} difficulty fill-in-the-blank question about {category}.
        
CRITICAL REQUIREMENTS:
- Must contain exactly one blank represented by '_____'
- Answer should be 1-3 words maximum
- Make it educational, clear, and meaningful
- Ensure the question is unique and not common
- The question should test actual knowledge, not be trivial
- Focus on important concepts in {category}
- Do NOT include any IDs, numbers, or placeholders in the question or answer

Return ONLY valid JSON with this exact structure:
{{
    "question": "the fill-in-the-blank question with exactly one _____",
    "answer": "the correct answer (1-3 words)",
    "explanation": "brief educational explanation"
}}

Example for English Language:
{{
    "question": "The author of 'Pride and Prejudice' is _____.",
    "answer": "Jane Austen",
    "explanation": "Jane Austen wrote the famous novel 'Pride and Prejudice' in 1813."
}}

Example for Quantitative Aptitude:
{{
    "question": "The square root of 144 is _____.",
    "answer": "12",
    "explanation": "12 multiplied by 12 equals 144, so the square root of 144 is 12."
}}"""
        
        return base_prompt

    def _validate_question(self, question_data, difficulty):
        required_keys = ['question', 'answer']
        
        for key in required_keys:
            if key not in question_data:
                raise ValueError(f"Missing required field: {key}")
        
        if '_____' not in question_data['question']:
            raise ValueError("Question must contain '_____' to represent the blank")
        
        if not question_data['question'].strip():
            raise ValueError("Question cannot be empty")
        
        if not question_data['answer'].strip():
            raise ValueError("Answer cannot be empty")
            
        # Count blanks to ensure exactly one
        blank_count = question_data['question'].count('_____')
        if blank_count != 1:
            raise ValueError(f"Question must contain exactly one blank, found {blank_count}")
        
        # Check for ID-like patterns in question or answer
        id_patterns = [r'concept_\d+', r'principle_\d+', r'id_\d+', r'\d+$']
        for pattern in id_patterns:
            if re.search(pattern, question_data['question'].lower()) or re.search(pattern, question_data['answer'].lower()):
                raise ValueError("Question or answer contains ID-like pattern")

    def _is_question_duplicate_in_current_quiz(self, question_text):
        """Check if question is duplicate in current quiz only"""
        return question_text in self.current_quiz_questions

    def _create_meaningful_fallback_question(self, difficulty, category):
        """Create meaningful fallback questions that don't use IDs"""
        # Define meaningful fallback questions for each category and difficulty
        fallback_questions = {
            'English Language': {
                'easy': [
                    {'question': 'The past tense of "run" is _____.', 'answer': 'ran', 'explanation': 'The irregular past tense of "run" is "ran".'},
                    {'question': 'A synonym for "happy" is _____.', 'answer': 'joyful', 'explanation': 'Joyful means feeling or expressing great happiness.'},
                    {'question': 'The plural of "child" is _____.', 'answer': 'children', 'explanation': 'The irregular plural form of "child" is "children".'},
                    {'question': 'The author of "Romeo and Juliet" is _____.', 'answer': 'Shakespeare', 'explanation': 'William Shakespeare wrote the famous play "Romeo and Juliet".'}
                ],
                'medium': [
                    {'question': 'The rhetorical device that compares two things using "like" or "as" is called _____.', 'answer': 'simile', 'explanation': 'A simile directly compares two different things using "like" or "as".'},
                    {'question': 'The literary term for the main character in a story is the _____.', 'answer': 'protagonist', 'explanation': 'The protagonist is the central character who drives the story forward.'},
                    {'question': 'Words that have opposite meanings are called _____.', 'answer': 'antonyms', 'explanation': 'Antonyms are words with opposite meanings, like hot and cold.'}
                ],
                'hard': [
                    {'question': 'The philosophical statement "I think, therefore I am" was coined by _____.', 'answer': 'Descartes', 'explanation': 'René Descartes established this foundational statement in Western philosophy.'},
                    {'question': 'The study of meaning in language is called _____.', 'answer': 'semantics', 'explanation': 'Semantics is the branch of linguistics concerned with meaning.'}
                ],
                'very hard': [
                    {'question': 'The logical paradox involving self-reference and truth is called the _____ paradox.', 'answer': 'liar', 'explanation': 'The liar paradox concerns statements that declare their own falsehood.'},
                    {'question': 'The branch of linguistics that studies sentence structure is called _____.', 'answer': 'syntax', 'explanation': 'Syntax is the study of how words combine to form grammatical sentences.'}
                ]
            },
            'Quantitative Aptitude': {
                'easy': [
                    {'question': '2 + 2 = _____.', 'answer': '4', 'explanation': 'Basic addition: 2 + 2 = 4.'},
                    {'question': 'The square root of 16 is _____.', 'answer': '4', 'explanation': '4 × 4 = 16, so √16 = 4.'}
                ],
                'medium': [
                    {'question': 'If x + 5 = 12, then x = _____.', 'answer': '7', 'explanation': 'Subtract 5 from both sides: x = 12 - 5 = 7.'},
                    {'question': '15% of 200 is _____.', 'answer': '30', 'explanation': '15% of 200 = 0.15 × 200 = 30.'}
                ],
                'hard': [
                    {'question': 'The derivative of x³ is _____.', 'answer': '3x²', 'explanation': 'Using the power rule: d/dx(x³) = 3x².'},
                    {'question': 'The probability of getting heads in a coin toss is _____.', 'answer': '1/2', 'explanation': 'A fair coin has equal probability for heads and tails.'}
                ],
                'very hard': [
                    {'question': 'The limit as x approaches 0 of (sin x)/x is _____.', 'answer': '1', 'explanation': 'This is a fundamental limit in calculus.'},
                    {'question': 'The imaginary unit i is defined as the square root of _____.', 'answer': '-1', 'explanation': 'i = √(-1), the fundamental imaginary unit.'}
                ]
            },
            'Reasoning Ability': {
                'easy': [
                    {'question': 'The next number in the sequence 2, 4, 6, 8 is _____.', 'answer': '10', 'explanation': 'This is an arithmetic sequence increasing by 2 each time.'},
                    {'question': 'If A = 1, B = 2, then C = _____.', 'answer': '3', 'explanation': 'The pattern is the position in the alphabet.'}
                ],
                'medium': [
                    {'question': 'The missing number in the pattern: 1, 1, 2, 3, 5, 8, _____.', 'answer': '13', 'explanation': 'This is the Fibonacci sequence where each number is the sum of the two preceding ones.'},
                    {'question': 'If yesterday was Monday, then tomorrow is _____.', 'answer': 'Wednesday', 'explanation': 'If yesterday was Monday, today is Tuesday, so tomorrow is Wednesday.'}
                ],
                'hard': [
                    {'question': 'In a group of 100 people, 70 like coffee, 80 like tea, and 60 like both. The number who like neither is _____.', 'answer': '10', 'explanation': 'Using inclusion-exclusion: 70 + 80 - 60 = 90 like at least one, so 100 - 90 = 10 like neither.'}
                ],
                'very hard': [
                    {'question': 'The minimum number of colors needed to color a map so no adjacent regions share the same color is _____.', 'answer': '4', 'explanation': 'This is the Four Color Theorem in mathematics.'}
                ]
            },
            'General/Banking Awareness': {
                'easy': [
                    {'question': 'The capital of India is _____.', 'answer': 'New Delhi', 'explanation': 'New Delhi is the capital city of India.'},
                    {'question': 'The currency of Japan is the _____.', 'answer': 'yen', 'explanation': 'The Japanese currency is called the yen.'}
                ],
                'medium': [
                    {'question': 'The World Bank headquarters is located in _____.', 'answer': 'Washington D.C.', 'explanation': 'The World Bank headquarters is in Washington D.C., USA.'},
                    {'question': 'The term for money that has no intrinsic value but is accepted by government decree is _____ money.', 'answer': 'fiat', 'explanation': 'Fiat money is established as money by government regulation.'}
                ],
                'hard': [
                    {'question': 'The Basel _____ accord introduced capital adequacy requirements.', 'answer': 'I', 'explanation': 'Basel I was the first Basel Accord dealing with capital requirements.'}
                ],
                'very hard': [
                    {'question': 'The economic theory advocating government spending to stimulate demand is associated with _____.', 'answer': 'Keynes', 'explanation': 'Keynesian economics emphasizes government intervention to manage economic cycles.'}
                ]
            },
            'Computer Aptitude': {
                'easy': [
                    {'question': 'The brain of a computer is the _____.', 'answer': 'CPU', 'explanation': 'The CPU (Central Processing Unit) is often called the brain of the computer.'},
                    {'question': 'HTML stands for _____ Markup Language.', 'answer': 'HyperText', 'explanation': 'HTML is HyperText Markup Language used for web pages.'}
                ],
                'medium': [
                    {'question': 'The protocol used for sending email is _____.', 'answer': 'SMTP', 'explanation': 'SMTP (Simple Mail Transfer Protocol) is used for sending emails.'},
                    {'question': 'The data structure that follows LIFO principle is called a _____.', 'answer': 'stack', 'explanation': 'Stack follows Last-In-First-Out (LIFO) principle.'}
                ],
                'hard': [
                    {'question': 'The complexity of binary search algorithm is O(_____).', 'answer': 'log n', 'explanation': 'Binary search has logarithmic time complexity O(log n).'}
                ],
                'very hard': [
                    {'question': 'The complexity class of problems that can be verified in polynomial time is _____.', 'answer': 'NP', 'explanation': 'NP (Nondeterministic Polynomial time) contains problems verifiable in polynomial time.'}
                ]
            }
        }
        
        # Get questions for the specific category and difficulty
        category_questions = fallback_questions.get(category, {})
        difficulty_questions = category_questions.get(difficulty, [])
        
        if difficulty_questions:
            # Pick a random question from the available ones
            selected = random.choice(difficulty_questions)
            
            # Check if this question is already in current quiz
            if selected['question'] in self.current_quiz_questions:
                # Try to find a different one
                for q in difficulty_questions:
                    if q['question'] not in self.current_quiz_questions:
                        selected = q
                        break
            
            fallback_question = {
                'question': selected['question'],
                'answer': selected['answer'],
                'category': category,
                'difficulty': difficulty,
                'explanation': selected['explanation'],
                'is_llm_generated': False,
                'question_type': 'fill_in_blank'
            }
            
            # Add to current quiz tracking
            self.current_quiz_questions.add(fallback_question['question'])
            
            print(f"📝 Using meaningful fallback question: {fallback_question['question'][:80]}...")
            return fallback_question
        
        # If no predefined questions available, create a simple meaningful one
        meaningful_fallback = {
            'question': f'An important concept in {category} is _____.',
            'answer': 'fundamental',
            'category': category,
            'difficulty': difficulty,
            'explanation': f'This question tests basic knowledge of {category} concepts.',
            'is_llm_generated': False,
            'question_type': 'fill_in_blank'
        }
        
        if not self._is_question_duplicate_in_current_quiz(meaningful_fallback['question']):
            self.current_quiz_questions.add(meaningful_fallback['question'])
        
        print(f"📝 Created meaningful fallback question for {difficulty} {category}")
        return meaningful_fallback

    def generate_fresh_question(self, difficulty, category, attempt_number=1):
        """Generate a completely fresh fill-in-the-blank question with optimized performance"""
        # Force fresh generation - don't use any cache
        print(f"🔄 FORCING FRESH GENERATION for {difficulty} {category} question...")
        
        if not self.api_available:
            print("⚠️ API not available, creating meaningful fallback question")
            return self._create_meaningful_fallback_question(difficulty, category)
        
        try:
            client = self._get_llm_client()
        except Exception as e:
            print(f"⚠️ Cannot initialize Groq client, using meaningful fallback: {str(e)}")
            return self._create_meaningful_fallback_question(difficulty, category)
        
        max_attempts = 8  # Increased attempts for better success rate
        
        for attempt in range(max_attempts):
            try:
                prompt = self._build_optimized_prompt(difficulty, category, attempt + 1)
                
                print(f"🔄 Generating FRESH {difficulty} {category} question (attempt {attempt + 1}/{max_attempts})...")
                
                # Use moderate temperature for balance between creativity and speed
                temperature = self._get_temperature(difficulty, attempt)
                
                # Add timeout and better error handling
                try:
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {
                                "role": "system", 
                                "content": """You are an EXPERT quiz master creating MEANINGFUL fill-in-the-blank questions. 
                                ALWAYS return valid JSON with exactly this structure:
                                {
                                    "question": "unique question text with exactly one _____",
                                    "answer": "correct answer (1-3 words)",
                                    "explanation": "brief educational explanation"
                                }
                                CRITICAL: Do NOT include any IDs, numbers, or placeholders in questions or answers.
                                Create ACTUAL educational questions that test real knowledge.
                                Do not include any other text or markdown formatting."""
                            },
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ],
                        temperature=temperature,
                        max_tokens=500,  # Increased for better responses
                        timeout=45  # Increased timeout
                    )
                except Exception as api_error:
                    print(f"❌ API error on attempt {attempt + 1}: {api_error}")
                    if attempt < max_attempts - 1:
                        time.sleep(2)
                        continue
                    else:
                        raise api_error
                
                content = response.choices[0].message.content.strip()
                print(f"📥 Received FRESH response for {difficulty} {category} question")
                
                # Extract and clean JSON from response
                try:
                    question_data = self._clean_json_response(content)
                except Exception as json_error:
                    print(f"❌ JSON parsing error: {json_error}")
                    print(f"Raw response: {content}")
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        raise json_error
                
                # Validate question structure
                try:
                    self._validate_question(question_data, difficulty)
                except Exception as validation_error:
                    print(f"❌ Question validation failed: {validation_error}")
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        raise validation_error
                
                # Create question object
                question = {
                    'question': question_data['question'],
                    'answer': question_data['answer'],
                    'category': category,
                    'difficulty': difficulty,
                    'explanation': question_data.get('explanation', 'No explanation provided.'),
                    'is_llm_generated': True,
                    'question_type': 'fill_in_blank'
                }
                
                # Check if question matches the intended difficulty (very lenient)
                if not self._validate_difficulty(question, difficulty):
                    print(f"⚠️ Question difficulty mismatch, but accepting anyway: {question['question'][:80]}...")
                
                # Check if question is duplicate in current quiz
                if self._is_question_duplicate_in_current_quiz(question['question']):
                    print(f"🔄 Question duplicate in current quiz, regenerating...")
                    continue
                
                # Check if question is similar to existing ones using vector DB (less strict)
                if self.vector_db.is_question_similar(question['question'], difficulty, similarity_threshold=0.85):
                    print(f"🔄 Question too similar to existing ones in database, regenerating...")
                    continue
                
                # Check if question is similar to current quiz questions (less strict)
                if self._is_question_similar_to_current_quiz(question['question'], similarity_threshold=0.8):
                    print(f"🔄 Question too similar to current quiz questions, regenerating...")
                    continue
                
                # Add to vector database (for future similarity checks)
                self.vector_db.add_question(question)
                
                # Add to current quiz tracking
                self.current_quiz_questions.add(question['question'])
                
                # Add embedding to current quiz embeddings for similarity checking
                embedding = self.vector_db._get_embedding(question['question'])
                self.current_quiz_embeddings.append(embedding)
                
                print(f"✅ Generated FRESH {difficulty} {category} question: {question['question'][:80]}...")
                return question
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    print("⚠️ All attempts failed, creating meaningful fallback question")
                    return self._create_meaningful_fallback_question(difficulty, category)
                
                # Wait before retry with reduced backoff
                time.sleep(min(1.5 ** attempt, 4))
        
        print("⚠️ Exhausted all attempts, creating meaningful fallback question")
        return self._create_meaningful_fallback_question(difficulty, category)

    def generate_questions_batch(self, difficulty, category, num_questions):
        """Generate multiple questions with NO CACHING"""
        questions = []
        
        print(f"🎯 Generating {num_questions} COMPLETELY FRESH questions for {difficulty} {category}...")
        
        # Generate questions with minimal delay between them
        for i in range(num_questions):
            print(f"🔧 Generating question {i+1}/{num_questions}...")
            question = self.generate_fresh_question(difficulty, category, i + 1)
            questions.append(question)
            
            # Small delay to avoid rate limiting
            if i < num_questions - 1:
                time.sleep(1.5)  # Slight delay between questions
        
        return questions

    def generate_dynamic_quiz_questions(self, difficulty, category, num_questions):
        """Generate a complete set of FRESH fill-in-the-blank questions WITH NO CACHING"""
        # Reset current quiz tracking for new quiz
        self.current_quiz_questions = set()
        self.current_quiz_embeddings = []
        
        print(f"🎯 Generating exactly {num_questions} COMPLETELY FRESH {difficulty} {category} questions...")
        print(f"🚫 CACHE DISABLED - All questions will be newly generated")
        
        # Generate new questions with NO CACHE
        questions = self.generate_questions_batch(difficulty, category, num_questions)
        
        print(f"🎉 Successfully prepared {len(questions)} COMPLETELY FRESH {difficulty} {category} questions!")
        return questions

    def generate_mixed_difficulty_quiz(self, category, num_questions):
        """Generate a quiz with questions from all difficulty levels - ALL FRESH"""
        # Reset current quiz tracking
        self.current_quiz_questions = set()
        self.current_quiz_embeddings = []
        
        print(f"🎯 Generating mixed difficulty quiz with exactly {num_questions} COMPLETELY FRESH {category} questions...")
        print(f"🚫 CACHE DISABLED - All questions will be newly generated")
        
        # Calculate distribution
        difficulties = ['easy', 'medium', 'hard', 'very hard']
        base_count = num_questions // len(difficulties)
        remainder = num_questions % len(difficulties)
        
        questions_per_difficulty = {}
        for i, difficulty in enumerate(difficulties):
            count = base_count
            if i < remainder:
                count += 1
            questions_per_difficulty[difficulty] = count
        
        print(f"📊 Question distribution: {questions_per_difficulty}")
        
        all_questions = []
        
        # Generate FRESH questions for each difficulty level
        for difficulty, count in questions_per_difficulty.items():
            if count > 0:
                print(f"🔧 Generating {count} COMPLETELY FRESH {difficulty} {category} questions...")
                try:
                    difficulty_questions = self.generate_dynamic_quiz_questions(difficulty, category, count)
                    all_questions.extend(difficulty_questions)
                    print(f"✅ Generated {len(difficulty_questions)} COMPLETELY FRESH {difficulty} {category} questions")
                except Exception as e:
                    print(f"❌ Failed to generate {difficulty} {category} questions: {e}")
                    # Create meaningful fallback questions
                    for i in range(count):
                        fallback_question = self._create_meaningful_fallback_question(difficulty, category)
                        if not self._is_question_duplicate_in_current_quiz(fallback_question['question']):
                            all_questions.append(fallback_question)
                            self.current_quiz_questions.add(fallback_question['question'])
                    print(f"📥 Created {count} meaningful fallback {difficulty} {category} questions")
        
        # Shuffle the questions to mix difficulties
        random.shuffle(all_questions)
        
        print(f"🎉 Mixed difficulty quiz ready with {len(all_questions)} COMPLETELY FRESH {category} questions!")
        return all_questions

class QuizManager:
    def __init__(self, database_path, question_generator):
        self.database_path = database_path
        self.question_generator = question_generator
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database tables if they don't exist"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                try:
                    cursor.execute('PRAGMA table_info(users)')
                    columns = [column[1] for column in cursor.fetchall()]
                    
                    # If password column is missing, drop and recreate the table
                    if 'password' not in columns:
                        print("Database schema mismatch detected. Fixing...")
                        cursor.execute('DROP TABLE IF EXISTS users')
                except Exception as e:
                    print(f"Schema check error: {e}")
                
                # Create tables
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS categories (
                        id INTEGER PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS quizzes (
                        id INTEGER PRIMARY KEY,
                        user_id INTEGER,
                        category_id INTEGER,
                        difficulty TEXT,
                        score INTEGER,
                        total_questions INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id),
                        FOREIGN KEY (category_id) REFERENCES categories(id)
                    )
                ''')
                
                # Insert ONLY the requested categories
                requested_categories = [
                    'English Language', 
                    'Quantitative Aptitude', 
                    'Reasoning Ability', 
                    'General/Banking Awareness', 
                    'Computer Aptitude'
                ]
                for category in requested_categories:
                    try:
                        cursor.execute('INSERT INTO categories (name) VALUES (?)', (category,))
                    except sqlite3.IntegrityError:
                        pass
                
                conn.commit()
                print("Database initialized successfully with requested categories")
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def get_categories(self):
        """Get all quiz categories - ONLY RETURN REQUESTED CATEGORIES"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT name FROM categories')
                all_categories = [row[0] for row in cursor.fetchall()]
                
                # Filter to only include the requested categories
                requested_categories = [
                    'English Language', 
                    'Quantitative Aptitude', 
                    'Reasoning Ability', 
                    'General/Banking Awareness', 
                    'Computer Aptitude'
                ]
                categories = [cat for cat in all_categories if cat in requested_categories]
                
                return categories if categories else ['English Language']
        except Exception as e:
            print(f"Error fetching categories: {e}")
            return ['English Language']
    
    def get_difficulties(self):
        """Get all difficulty levels"""
        return ['easy', 'medium', 'hard', 'very hard', 'mixed']
    
    def get_question_numbers(self):
        """Get available number of questions options"""
        return [5, 10, 15, 20]
    
    def get_user_statistics(self, user_id):
        """Get user quiz statistics"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) as total_quizzes, MAX(score) as best_score
                    FROM quizzes
                    WHERE user_id = ?
                ''', (user_id,))
                
                result = cursor.fetchone()
                if result and result[0] > 0:
                    return {
                        'total_quizzes': result[0],
                        'best_score': result[1]
                    }
                return None
        except Exception as e:
            print(f"Error fetching user statistics: {e}")
            return None

    def get_user_performance_data(self, user_id):
        """Get detailed performance data for the user"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Get overall statistics
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_quizzes,
                        SUM(score) as total_correct,
                        SUM(total_questions) as total_questions,
                        MAX(score) as best_score,
                        AVG(score * 100.0 / total_questions) as average_percentage
                    FROM quizzes 
                    WHERE user_id = ?
                ''', (user_id,))
                overall_stats = cursor.fetchone()
                
                # Get quiz history
                cursor.execute('''
                    SELECT 
                        q.created_at,
                        c.name as category,
                        q.difficulty,
                        q.score,
                        q.total_questions,
                        ROUND((q.score * 100.0 / q.total_questions), 1) as percentage
                    FROM quizzes q
                    JOIN categories c ON q.category_id = c.id
                    WHERE q.user_id = ?
                    ORDER BY q.created_at DESC
                    LIMIT 10
                ''', (user_id,))
                quiz_history = cursor.fetchall()
                
                # Get performance by category
                cursor.execute('''
                    SELECT 
                        c.name as category,
                        COUNT(*) as quiz_count,
                        AVG(q.score * 100.0 / q.total_questions) as average_score
                    FROM quizzes q
                    JOIN categories c ON q.category_id = c.id
                    WHERE q.user_id = ?
                    GROUP BY c.name
                    ORDER BY quiz_count DESC
                ''', (user_id,))
                category_stats = cursor.fetchall()
                
                # Get performance by difficulty
                cursor.execute('''
                    SELECT 
                        difficulty,
                        COUNT(*) as quiz_count,
                        AVG(score * 100.0 / total_questions) as average_score
                    FROM quizzes
                    WHERE user_id = ?
                    GROUP BY difficulty
                    ORDER BY quiz_count DESC
                ''', (user_id,))
                difficulty_stats = cursor.fetchall()
                
                # Format the data
                performance_data = {
                    'overall': {
                        'total_quizzes': overall_stats[0] if overall_stats else 0,
                        'total_correct': overall_stats[1] if overall_stats else 0,
                        'total_questions': overall_stats[2] if overall_stats else 0,
                        'best_score': overall_stats[3] if overall_stats else 0,
                        'average_percentage': round(overall_stats[4], 1) if overall_stats and overall_stats[4] else 0,
                        'accuracy': round((overall_stats[1] * 100.0 / overall_stats[2]), 1) if overall_stats and overall_stats[2] and overall_stats[2] > 0 else 0
                    },
                    'quiz_history': [
                        {
                            'date': row[0],
                            'category': row[1],
                            'difficulty': row[2],
                            'score': row[3],
                            'total_questions': row[4],
                            'percentage': row[5]
                        } for row in quiz_history
                    ],
                    'category_stats': [
                        {
                            'category': row[0],
                            'quiz_count': row[1],
                            'average_score': round(row[2], 1) if row[2] else 0
                        } for row in category_stats
                    ],
                    'difficulty_stats': [
                        {
                            'difficulty': row[0],
                            'quiz_count': row[1],
                            'average_score': round(row[2], 1) if row[2] else 0
                        } for row in difficulty_stats
                    ]
                }
                
                return performance_data
                
        except Exception as e:
            print(f"Error fetching performance data: {e}")
            return None

# ========== TEMPLATE STRINGS ==========

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}QuizMaster Pro{% endblock %}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: #333; }
        nav { background-color: rgba(0, 0, 0, 0.8); padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3); }
        nav a { color: white; text-decoration: none; margin: 0 1rem; transition: color 0.3s; }
        nav a:hover { color: #667eea; }
        .nav-brand { font-size: 1.5rem; font-weight: bold; color: #667eea; }
        .nav-links { display: flex; gap: 2rem; align-items: center; }
        .container { max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }
        .flash-messages { margin-bottom: 2rem; }
        .alert { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
        .alert-success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .alert-error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .content { background: white; border-radius: 1rem; padding: 2rem; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); min-height: 500px; }
        footer { text-align: center; color: white; margin-top: 3rem; padding-bottom: 2rem; }
        .user-welcome { color: #667eea; font-weight: bold; }
    </style>
    {% block extra_css %}{% endblock %}
</head>
<body>
    <nav>
        <span class="nav-brand">QuizMaster Pro</span>
        <div class="nav-links">
            <a href="{{ url_for('index') }}">Home</a>
            {% if "user_id" in session %}
                <a href="{{ url_for('performance') }}">Performance</a>
                <span class="user-welcome">Welcome, {{ session.username }}!</span>
                <a href="{{ url_for('logout') }}">Logout</a>
            {% else %}
                <a href="{{ url_for('login') }}">Login</a>
                <a href="{{ url_for('register') }}">Register</a>
            {% endif %}
        </div>
    </nav>
    
    <div class="container">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <div class="flash-messages">
                    {% for category, message in messages %}
                        <div class="alert alert-{{ category }}">{{ message }}</div>
                    {% endfor %}
                </div>
            {% endif %}
        {% endwith %}
        
        <div class="content">
            {% block content %}{% endblock %}
        </div>
    </div>
    
    <footer>
        <p>&copy; 2025 QuizMaster Pro. All rights reserved.</p>
    </footer>
</body>
</html>
'''

INDEX_TEMPLATE = '''
{% extends "base.html" %}

{% block title %}Home - QuizMaster Pro{% endblock %}

{% block content %}
<style>
    .api-status-active { color: #28a745; font-weight: bold; }
    .api-status-inactive { color: #dc3545; font-weight: bold; }
    .hero-section { text-align: center; padding: 2rem 0; }
    .stats-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 2rem 0; }
    .stat-card { background: white; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
    .stat-value { font-size: 2rem; font-weight: bold; color: #667eea; margin-bottom: 0.5rem; }
    .stat-label { color: #666; font-size: 0.9rem; }
    .quiz-section { background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; border: 2px solid #e9ecef; }
    .form-group { margin-bottom: 1.5rem; }
    .form-label { display: block; margin-bottom: 0.5rem; font-weight: bold; color: #333; }
    .form-select { width: 100%; padding: 0.75rem; border: 1px solid #ddd; border-radius: 0.3rem; font-size: 1rem; background: white; }
    .btn-start-quiz { display: block; width: 100%; padding: 1rem 2rem; background: #667eea; color: white; border: none; border-radius: 0.5rem; font-size: 1.2rem; font-weight: bold; cursor: pointer; transition: all 0.3s; }
    .btn-start-quiz:hover { background: #5568d3; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); }
    .login-prompt { background: #e8f4f8; padding: 2rem; border-radius: 0.5rem; text-align: center; border-left: 4px solid #667eea; }
    .btn-login, .btn-register { display: inline-block; padding: 0.75rem 1.5rem; margin: 0 0.5rem; border-radius: 0.3rem; font-weight: bold; text-decoration: none; transition: all 0.3s; }
    .btn-login { background: #667eea; color: white; }
    .btn-register { background: #764ba2; color: white; }
    .btn-login:hover, .btn-register:hover { transform: translateY(-2px); }
    .quiz-form-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2rem; margin-bottom: 2rem; }
    @media (max-width: 768px) {
        .quiz-form-grid { grid-template-columns: 1fr; }
    }
    .same-settings-section { background: #e8f4f8; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; border: 2px solid #667eea; }
</style>

<div class="hero-section">
    <h1 style="color: #667eea; margin-bottom: 1rem;">Welcome to QuizMaster Pro</h1>
    <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Test your knowledge with AI-powered fill-in-the-blank quizzes</p>
    
    <!-- Stats Section -->
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-value">{{ stats.user_quizzes }}</div>
            <div class="stat-label">Total Quizzes</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{{ stats.best_score }}</div>
            <div class="stat-label">Best Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{{ stats.total_categories }}</div>
            <div class="stat-label">Categories</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">AI Status</div>
            <div class="{% if stats.api_status == "ACTIVE" %}api-status-active{% else %}api-status-inactive{% endif %}">
                {{ stats.api_status }}
            </div>
        </div>
    </div>
    
    <!-- Vector DB Stats -->
    {% if stats.vector_db_stats %}
    <div style="background: #e8f4f8; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; text-align: center;">
        <h3 style="color: #667eea; margin-bottom: 0.5rem;">📊 Vector Database Stats</h3>
        <p style="color: #666; margin: 0;">
            Total Questions: {{ stats.vector_db_stats.total_questions }} |
            Easy: {{ stats.vector_db_stats.easy }} |
            Medium: {{ stats.vector_db_stats.medium }} |
            Hard: {{ stats.vector_db_stats.hard }} |
            Very Hard: {{ stats.vector_db_stats.very_hard }}
        </p>
    </div>
    {% endif %}
    
    <!-- API Status -->
    <div style="background: #e8f4f8; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; text-align: center;">
        <h3 style="color: #667eea; margin-bottom: 0.5rem;">🤖 AI-Powered Fresh Questions</h3>
        <p style="color: #666; margin: 0;">{{ stats.api_message }}</p>
        <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            Every quiz contains COMPLETELY NEW questions with NO CACHING for maximum freshness
        </p>
    </div>
    
    <!-- Quick Restart Section -->
    {% if "user_id" in session and session.get('quiz_category') %}
    <div class="same-settings-section">
        <h2 style="color: #667eea; text-align: center; margin-bottom: 1rem;">Quick Restart with Same Settings</h2>
        <p style="text-align: center; color: #666; margin-bottom: 1.5rem;">
            Start a new quiz with the same category and difficulty, but with COMPLETELY NEW questions!
        </p>
        <div style="text-align: center;">
            <form method="POST" action="{{ url_for('start_same_quiz') }}" style="display: inline-block;">
                <button type="submit" class="btn-start-quiz" style="background: #764ba2;">
                    🔄 Start New {{ session.get('quiz_difficulty', 'Mixed')|title }} {{ session.get('quiz_category', 'English Language') }} Quiz
                </button>
            </form>
            <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #666;">
                Previous: {{ session.get('num_questions', 5) }} questions • {{ session.get('quiz_difficulty', 'mixed')|title }}
            </div>
        </div>
    </div>
    {% endif %}
    
    <!-- Quiz Section - Only for logged in users -->
    {% if "user_id" in session %}
        <div class="quiz-section">            
            <form method="POST" action="{{ url_for('start_quiz') }}">
                <div class="quiz-form-grid">
                    <div class="form-group">
                        <label class="form-label">📚 Category</label>
                        <select name="category" class="form-select" required>
                            {% for category in categories %}
                                <option value="{{ category }}">{{ category }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">🎯 Difficulty</label>
                        <select name="difficulty" class="form-select" required>
                            {% for difficulty in difficulties %}
                                <option value="{{ difficulty }}">{{ difficulty|title }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">🔢 Number of Questions</label>
                        <select name="num_questions" class="form-select" required>
                            {% for num in question_numbers %}
                                <option value="{{ num }}">{{ num }} Questions</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
                
                <button type="submit" class="btn-start-quiz">
                    🚀 Start Fresh Quiz (No Repeated Questions)
                </button>
            </form>
        </div>
    {% else %}
        <!-- Login Prompt for non-logged in users -->
        <div class="login-prompt">
            <h2 style="color: #667eea; margin-bottom: 1rem;">Ready to Test Your Knowledge?</h2>
            <p style="color: #666; margin-bottom: 2rem; font-size: 1.1rem;">
                Log in or create an account to start taking AI-powered fill-in-the-blank quizzes with COMPLETELY NEW questions every time!
            </p>
            <div>
                <a href="{{ url_for('login') }}" class="btn-login">Login</a>
                <a href="{{ url_for('register') }}" class="btn-register">Register</a>
            </div>
        </div>
    {% endif %}
</div>
{% endblock %}
'''

LOGIN_TEMPLATE = '''
{% extends "base.html" %}

{% block title %}Login - QuizMaster Pro{% endblock %}

{% block content %}
<style>
    .auth-container { max-width: 400px; margin: 0 auto; }
    .auth-form { background: white; padding: 2rem; border-radius: 0.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    .form-group { margin-bottom: 1.5rem; }
    .form-label { display: block; margin-bottom: 0.5rem; font-weight: bold; color: #333; }
    .form-input { width: 100%; padding: 0.75rem; border: 1px solid #ddd; border-radius: 0.3rem; font-size: 1rem; }
    .btn-submit { width: 100%; padding: 0.75rem; background: #667eea; color: white; border: none; border-radius: 0.3rem; font-size: 1.1rem; font-weight: bold; cursor: pointer; transition: background 0.3s; }
    .btn-submit:hover { background: #5568d3; }
    .auth-links { text-align: center; margin-top: 1.5rem; }
    .auth-links a { color: #667eea; text-decoration: none; }
</style>

<div class="auth-container">
    <h1 style="text-align: center; color: #667eea; margin-bottom: 2rem;">Login to QuizMaster Pro</h1>
    
    <div class="auth-form">
        <form method="POST" action="{{ url_for('login') }}">
            <div class="form-group">
                <label class="form-label">Username</label>
                <input type="text" name="username" class="form-input" required>
            </div>
            
            <div class="form-group">
                <label class="form-label">Password</label>
                <input type="password" name="password" class="form-input" required>
            </div>
            
            <button type="submit" class="btn-submit">Login</button>
        </form>
        
        <div class="auth-links">
            <p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p>
        </div>
    </div>
</div>
{% endblock %}
'''

REGISTER_TEMPLATE = '''
{% extends "base.html" %}

{% block title %}Register - QuizMaster Pro{% endblock %}

{% block content %}
<style>
    .auth-container { max-width: 400px; margin: 0 auto; }
    .auth-form { background: white; padding: 2rem; border-radius: 0.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    .form-group { margin-bottom: 1.5rem; }
    .form-label { display: block; margin-bottom: 0.5rem; font-weight: bold; color: #333; }
    .form-input { width: 100%; padding: 0.75rem; border: 1px solid #ddd; border-radius: 0.3rem; font-size: 1rem; }
    .btn-submit { width: 100%; padding: 0.75rem; background: #667eea; color: white; border: none; border-radius: 0.3rem; font-size: 1.1rem; font-weight: bold; cursor: pointer; transition: background 0.3s; }
    .btn-submit:hover { background: #5568d3; }
    .auth-links { text-align: center; margin-top: 1.5rem; }
    .auth-links a { color: #667eea; text-decoration: none; }
</style>

<div class="auth-container">
    <h1 style="text-align: center; color: #667eea; margin-bottom: 2rem;">Create Account</h1>
    
    <div class="auth-form">
        <form method="POST" action="{{ url_for('register') }}">
            <div class="form-group">
                <label class="form-label">Username</label>
                <input type="text" name="username" class="form-input" required>
            </div>
            
            <div class="form-group">
                <label class="form-label">Password</label>
                <input type="password" name="password" class="form-input" required minlength="6">
                <small style="color: #666;">Password must be at least 6 characters long</small>
            </div>
            
            <div class="form-group">
                <label class="form-label">Confirm Password</label>
                <input type="password" name="confirm_password" class="form-input" required>
            </div>
            
            <button type="submit" class="btn-submit">Create Account</button>
        </form>
        
        <div class="auth-links">
            <p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
        </div>
    </div>
</div>
{% endblock %}
'''

QUIZ_TEMPLATE = '''
{% extends "base.html" %}

{% block title %}Quiz - QuizMaster Pro{% endblock %}

{% block content %}
<style>
    .quiz-container { max-width: 800px; margin: 0 auto; }
    .quiz-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 2px solid #e9ecef; }
    .quiz-progress { background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem; }
    .progress-bar { width: 100%; height: 10px; background: #e9ecef; border-radius: 5px; overflow: hidden; }
    .progress-fill { height: 100%; background: #667eea; transition: width 0.3s; }
    .question-card { background: white; padding: 2rem; border-radius: 0.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 2rem; }
    .question-text { font-size: 1.3rem; margin-bottom: 1.5rem; color: #333; line-height: 1.6; }
    .answer-form .form-group { margin-bottom: 1.5rem; }
    .answer-input { width: 100%; padding: 1rem; border: 2px solid #ddd; border-radius: 0.5rem; font-size: 1.1rem; transition: border-color 0.3s; }
    .answer-input:focus { border-color: #667eea; outline: none; }
    .btn-submit { padding: 1rem 2rem; background: #667eea; color: white; border: none; border-radius: 0.5rem; font-size: 1.1rem; font-weight: bold; cursor: pointer; transition: all 0.3s; }
    .btn-submit:hover { background: #5568d3; transform: translateY(-2px); }
    .difficulty-badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 1rem; font-size: 0.8rem; font-weight: bold; margin-left: 1rem; }
    .difficulty-easy { background: #d4edda; color: #155724; }
    .difficulty-medium { background: #fff3cd; color: #856404; }
    .difficulty-hard { background: #f8d7da; color: #721c24; }
    .difficulty-very-hard { background: #d1ecf1; color: #0c5460; }
</style>

<div class="quiz-container">
    <div class="quiz-header">
        <h1 style="color: #667eea;">Fill-in-the-Blank Quiz</h1>
        <div style="color: #666;">
            Question <strong>{{ question_number }}</strong> of <strong>{{ total_questions }}</strong>
        </div>
    </div>
    
    <div class="quiz-progress">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Progress</span>
            <span>{{ ((question_number / total_questions) * 100)|round|int }}%</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {{ (question_number / total_questions) * 100 }}%;"></div>
        </div>
    </div>
    
    <div class="question-card">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <h2 style="color: #333; margin: 0;">Question</h2>
            <span class="difficulty-badge difficulty-{{ question.difficulty|replace(' ', '-') }}">
                {{ question.difficulty|title }}
            </span>
        </div>
        
        <div class="question-text">
            {{ question.question|replace('_____', '<span style="border-bottom: 2px dashed #667eea; padding: 0 0.5rem;">__________</span>')|safe }}
        </div>
        
        <form method="POST" action="{{ url_for('submit_answer') }}" class="answer-form">
            <div class="form-group">
                <label for="answer" style="display: block; margin-bottom: 0.5rem; font-weight: bold; color: #333;">
                    Your Answer:
                </label>
                <input type="text" name="answer" id="answer" class="answer-input" required autofocus 
                       placeholder="Type your answer here...">
            </div>
            
            <button type="submit" class="btn-submit">
                {% if question_number == total_questions %}
                    Finish Quiz
                {% else %}
                    Submit Answer & Continue
                {% endif %}
            </button>
        </form>
    </div>
    
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>💡 Tip: Be specific with your answers. The system will check for variations and synonyms.</p>
    </div>
</div>
{% endblock %}
'''

QUIZ_RESULTS_TEMPLATE = '''
{% extends "base.html" %}

{% block title %}Quiz Results - QuizMaster Pro{% endblock %}

{% block content %}
<style>
    .results-container { max-width: 900px; margin: 0 auto; }
    .results-header { text-align: center; margin-bottom: 3rem; }
    .score-circle { width: 150px; height: 150px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; display: flex; flex-direction: column; justify-content: center; align-items: center; margin: 0 auto 2rem; }
    .score-percentage { font-size: 2.5rem; font-weight: bold; }
    .score-text { font-size: 1.1rem; }
    .results-summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 3rem; }
    .summary-card { background: white; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
    .summary-value { font-size: 2rem; font-weight: bold; color: #667eea; margin-bottom: 0.5rem; }
    .summary-label { color: #666; }
    .question-review { margin-top: 2rem; }
    .question-item { background: white; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 1.5rem; }
    .question-header { display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem; }
    .question-text { font-size: 1.1rem; margin-bottom: 1rem; line-height: 1.6; }
    .answer-comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
    .answer-box { padding: 1rem; border-radius: 0.5rem; }
    .user-answer { background: #f8f9fa; border-left: 4px solid #dc3545; }
    .correct-answer { background: #f8f9fa; border-left: 4px solid #28a745; }
    .answer-label { font-weight: bold; margin-bottom: 0.5rem; }
    .correct { border-left-color: #28a745; background: #d4edda; }
    .incorrect { border-left-color: #dc3545; background: #f8d7da; }
    .explanation { background: #e8f4f8; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem; }
    .actions { text-align: center; margin-top: 3rem; }
    .btn-action { display: inline-block; padding: 1rem 2rem; margin: 0 0.5rem; background: #667eea; color: white; text-decoration: none; border-radius: 0.5rem; font-weight: bold; transition: all 0.3s; }
    .btn-action:hover { background: #5568d3; transform: translateY(-2px); }
</style>

<div class="results-container">
    <div class="results-header">
        <h1 style="color: #667eea; margin-bottom: 1rem;">Quiz Completed!</h1>
        <div class="score-circle">
            <div class="score-percentage">{{ percentage }}%</div>
            <div class="score-text">{{ score }}/{{ total }} Correct</div>
        </div>
        
        <div class="results-summary">
            <div class="summary-card">
                <div class="summary-value">{{ total }}</div>
                <div class="summary-label">Total Questions</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{{ score }}</div>
                <div class="summary-label">Correct Answers</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{{ total - score }}</div>
                <div class="summary-label">Incorrect Answers</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{{ percentage }}%</div>
                <div class="summary-label">Success Rate</div>
            </div>
        </div>
    </div>
    
    <div class="question-review">
        <h2 style="color: #333; margin-bottom: 1.5rem; text-align: center;">Question Review</h2>
        
        {% for i in range(questions|length) %}
            {% set question = questions[i] %}
            {% set user_answer = user_answers.get(i|string, "Not answered") %}
            {% set is_correct = is_answer_correct(user_answer, question.answer) %}
            
            <div class="question-item {% if is_correct %}correct{% else %}incorrect{% endif %}">
                <div class="question-header">
                    <h3 style="margin: 0; color: #333;">Question {{ i + 1 }}</h3>
                    <span style="font-weight: bold; {% if is_correct %}color: #28a745;{% else %}color: #dc3545;{% endif %}">
                        {% if is_correct %}✅ Correct{% else %}❌ Incorrect{% endif %}
                    </span>
                </div>
                
                <div class="question-text">
                    {{ question.question|replace('_____', '<span style="background: #ffeb3b; padding: 0.2rem 0.5rem; border-radius: 0.3rem; font-weight: bold;">' + question.answer + '</span>')|safe }}
                </div>
                
                <div class="answer-comparison">
                    <div class="answer-box user-answer">
                        <div class="answer-label">Your Answer:</div>
                        <div>{{ user_answer if user_answer else "Not answered" }}</div>
                    </div>
                    <div class="answer-box correct-answer">
                        <div class="answer-label">Correct Answer:</div>
                        <div>{{ question.answer }}</div>
                    </div>
                </div>
                
                {% if question.explanation and question.explanation != "No explanation provided." %}
                    <div class="explanation">
                        <strong>Explanation:</strong> {{ question.explanation }}
                    </div>
                {% endif %}
            </div>
        {% endfor %}
    </div>
    
    <div class="actions">
        <a href="{{ url_for('index') }}" class="btn-action">🏠 Back to Home</a>
        <a href="{{ url_for('restart_quiz_same_settings') }}" class="btn-action" style="background: #764ba2;">
            🔄 New Quiz (Same Settings - Fresh Questions)
        </a>
    </div>
</div>
{% endblock %}
'''

PERFORMANCE_TEMPLATE = '''
{% extends "base.html" %}

{% block title %}Performance - QuizMaster Pro{% endblock %}

{% block content %}
<style>
    .performance-container { max-width: 1200px; margin: 0 auto; }
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 3rem; }
    .stat-card { background: white; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
    .stat-value { font-size: 2.5rem; font-weight: bold; color: #667eea; margin-bottom: 0.5rem; }
    .stat-label { color: #666; font-size: 1rem; }
    .section { background: white; padding: 2rem; border-radius: 0.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 2rem; }
    .section-title { color: #667eea; margin-bottom: 1.5rem; padding-bottom: 0.5rem; border-bottom: 2px solid #e9ecef; }
    .table { width: 100%; border-collapse: collapse; }
    .table th, .table td { padding: 1rem; text-align: left; border-bottom: 1px solid #e9ecef; }
    .table th { background: #f8f9fa; font-weight: bold; color: #333; }
    .progress-bar { width: 100%; height: 10px; background: #e9ecef; border-radius: 5px; overflow: hidden; }
    .progress-fill { height: 100%; background: #667eea; }
    .good { color: #28a745; }
    .average { color: #ffc107; }
    .poor { color: #dc3545; }
</style>

<div class="performance-container">
    <h1 style="color: #667eea; text-align: center; margin-bottom: 2rem;">Your Performance Dashboard</h1>
    
    {% if performance_data %}
        <!-- Overall Statistics -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ performance_data.overall.total_quizzes }}</div>
                <div class="stat-label">Total Quizzes Taken</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ performance_data.overall.best_score }}</div>
                <div class="stat-label">Best Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ performance_data.overall.average_percentage }}%</div>
                <div class="stat-label">Average Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ performance_data.overall.accuracy }}%</div>
                <div class="stat-label">Overall Accuracy</div>
            </div>
        </div>
        
        <!-- Quiz History -->
        <div class="section">
            <h2 class="section-title">Recent Quiz History</h2>
            {% if performance_data.quiz_history %}
                <table class="table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Category</th>
                            <th>Difficulty</th>
                            <th>Score</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for quiz in performance_data.quiz_history %}
                            <tr>
                                <td>{{ quiz.date[:16] }}</td>
                                <td>{{ quiz.category }}</td>
                                <td>{{ quiz.difficulty|title }}</td>
                                <td>{{ quiz.score }}/{{ quiz.total_questions }}</td>
                                <td class="{% if quiz.percentage >= 80 %}good{% elif quiz.percentage >= 60 %}average{% else %}poor{% endif %}">
                                    {{ quiz.percentage }}%
                                </td>
                            </tr>
                        {% endfor %}
                    </tbody>
                </table>
            {% else %}
                <p style="text-align: center; color: #666; padding: 2rem;">No quiz history yet. Take your first quiz to see your performance data!</p>
            {% endif %}
        </div>
        
        <!-- Performance by Category -->
        <div class="section">
            <h2 class="section-title">Performance by Category</h2>
            {% if performance_data.category_stats %}
                <table class="table">
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Quizzes Taken</th>
                            <th>Average Score</th>
                            <th>Performance</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for stat in performance_data.category_stats %}
                            <tr>
                                <td>{{ stat.category }}</td>
                                <td>{{ stat.quiz_count }}</td>
                                <td class="{% if stat.average_score >= 80 %}good{% elif stat.average_score >= 60 %}average{% else %}poor{% endif %}">
                                    {{ stat.average_score }}%
                                </td>
                                <td>
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: {{ stat.average_score }}%;"></div>
                                    </div>
                                </td>
                            </tr>
                        {% endfor %}
                    </tbody>
                </table>
            {% else %}
                <p style="text-align: center; color: #666; padding: 2rem;">No category data available yet.</p>
            {% endif %}
        </div>
        
        <!-- Performance by Difficulty -->
        <div class="section">
            <h2 class="section-title">Performance by Difficulty</h2>
            {% if performance_data.difficulty_stats %}
                <table class="table">
                    <thead>
                        <tr>
                            <th>Difficulty</th>
                            <th>Quizzes Taken</th>
                            <th>Average Score</th>
                            <th>Performance</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for stat in performance_data.difficulty_stats %}
                            <tr>
                                <td>{{ stat.difficulty|title }}</td>
                                <td>{{ stat.quiz_count }}</td>
                                <td class="{% if stat.average_score >= 80 %}good{% elif stat.average_score >= 60 %}average{% else %}poor{% endif %}">
                                    {{ stat.average_score }}%
                                </td>
                                <td>
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: {{ stat.average_score }}%;"></div>
                                    </div>
                                </td>
                            </tr>
                        {% endfor %}
                    </tbody>
                </table>
            {% else %}
                <p style="text-align: center; color: #666; padding: 2rem;">No difficulty data available yet.</p>
            {% endif %}
        </div>
        
    {% else %}
        <div style="text-align: center; padding: 3rem; color: #666;">
            <h2>No Performance Data Yet</h2>
            <p>Take your first quiz to start tracking your performance!</p>
            <a href="{{ url_for('index') }}" style="display: inline-block; margin-top: 1rem; padding: 1rem 2rem; background: #667eea; color: white; text-decoration: none; border-radius: 0.5rem;">Start Your First Quiz</a>
        </div>
    {% endif %}
</div>
{% endblock %}
'''

# Create instances after all classes are defined
question_generator = DynamicQuestionGenerator(API_AVAILABLE, API_KEY)
quiz_manager = QuizManager(app.config['DATABASE'], question_generator)

# ========== ROUTES ==========

@app.route('/')
def index():
    categories = quiz_manager.get_categories()
    difficulties = quiz_manager.get_difficulties()
    question_numbers = quiz_manager.get_question_numbers()
    
    api_status = 'ACTIVE' if API_AVAILABLE else 'INACTIVE'
    api_message = ''
    
    if API_AVAILABLE and API_KEY:
        api_message = 'Groq LLM is ready to generate COMPLETELY NEW questions with NO CACHING'
    elif not API_KEY:
        api_message = 'GROQ_API_KEY environment variable not set'
    else:
        api_message = 'Groq client initialization failed'
    
    # Get vector DB statistics
    vector_db_stats = {}
    if hasattr(question_generator, 'vector_db'):
        total_questions = question_generator.vector_db.get_question_count()
        vector_db_stats = {
            'total_questions': total_questions,
            'easy': question_generator.vector_db.get_question_count('easy'),
            'medium': question_generator.vector_db.get_question_count('medium'),
            'hard': question_generator.vector_db.get_question_count('hard'),
            'very_hard': question_generator.vector_db.get_question_count('very hard')
        }
    
    stats = {
        'total_questions': f"{vector_db_stats.get('total_questions', 0)} (Vector DB)",
        'total_categories': len(categories),
        'user_quizzes': 0,
        'best_score': 0,
        'api_status': f'{api_status}',
        'api_message': api_message,
        'vector_db_stats': vector_db_stats
    }
    
    if 'user_id' in session:
        user_stats = quiz_manager.get_user_statistics(session['user_id'])
        if user_stats:
            stats['user_quizzes'] = user_stats['total_quizzes']
            stats['best_score'] = user_stats['best_score']
    
    return render_template_string(INDEX_TEMPLATE, 
                         categories=categories, 
                         difficulties=difficulties,
                         question_numbers=question_numbers,
                         stats=stats,
                         api_available=API_AVAILABLE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return redirect(url_for('login'))
        
        try:
            with sqlite3.connect(app.config['DATABASE']) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, password FROM users WHERE username = ?', (username,))
                user = cursor.fetchone()
                
                if user:
                    user_id, stored_password_hash = user
                    input_password_hash = hash_password(password)
                    
                    if stored_password_hash == input_password_hash:
                        session['user_id'] = user_id
                        session['username'] = username
                        flash(f'Welcome back, {username}!', 'success')
                        return redirect(url_for('index'))
                    else:
                        flash('Invalid username or password', 'error')
                else:
                    flash('Invalid username or password', 'error')
        except Exception as e:
            print(f"Login error: {e}")
            flash('An error occurred during login', 'error')
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return redirect(url_for('register'))
        
        try:
            with sqlite3.connect(app.config['DATABASE']) as conn:
                cursor = conn.cursor()
                password_hash = hash_password(password)
                cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                             (username, password_hash))
                conn.commit()
                flash('Account created successfully! Please log in.', 'success')
                return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists', 'error')
        except Exception as e:
            print(f"Registration error: {e}")
            flash('An error occurred during registration', 'error')
    
    return render_template_string(REGISTER_TEMPLATE)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('index'))

@app.route('/performance')
@login_required
def performance():
    """Performance dashboard showing user statistics and quiz history"""
    performance_data = quiz_manager.get_user_performance_data(session['user_id'])
    return render_template_string(PERFORMANCE_TEMPLATE, performance_data=performance_data)

@app.route('/start-quiz', methods=['POST'])
@login_required
def start_quiz():
    # Clear any previous quiz session completely
    session_keys = ['quiz_questions', 'current_question', 'quiz_score', 'quiz_category', 'quiz_difficulty', 'user_answers', 'quiz_stored', 'num_questions']
    for key in session_keys:
        session.pop(key, None)
    
    # Also reset the generator's current quiz tracking
    if hasattr(question_generator, 'current_quiz_questions'):
        question_generator.current_quiz_questions = set()
        question_generator.current_quiz_embeddings = []
    
    category = request.form.get('category', 'English Language')
    difficulty = request.form.get('difficulty', 'mixed')
    num_questions = int(request.form.get('num_questions', 5))
    
    try:
        # Generate COMPLETELY FRESH questions based on selected difficulty
        if difficulty == 'mixed':
            questions = question_generator.generate_mixed_difficulty_quiz(category, num_questions)
            difficulty_display = 'Mixed (All Levels)'
        else:
            questions = question_generator.generate_dynamic_quiz_questions(difficulty, category, num_questions)
            difficulty_display = difficulty.title()
        
        if not questions:
            flash('Could not generate any questions. Please try again with different settings.', 'error')
            return redirect(url_for('index'))
        
        # Store quiz session
        session['quiz_questions'] = questions
        session['current_question'] = 0
        session['quiz_score'] = 0
        session['quiz_category'] = category
        session['quiz_difficulty'] = difficulty
        session['num_questions'] = num_questions
        session['user_answers'] = {}
        session['quiz_stored'] = False
        
        print(f"🎯 {difficulty_display} {category} quiz started with {len(questions)} COMPLETELY FRESH questions!")
        flash(f'{difficulty_display} {category} quiz started with {len(questions)} COMPLETELY NEW questions!', 'success')
        return redirect(url_for('take_quiz'))
    except Exception as e:
        print(f"Error generating quiz: {e}")
        flash(f'Error generating quiz: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/start-same-quiz', methods=['POST'])
@login_required
def start_same_quiz():
    """Start a new quiz with same category/difficulty but COMPLETELY FRESH questions"""
    # Get previous quiz settings from session
    category = session.get('quiz_category', 'English Language')
    difficulty = session.get('quiz_difficulty', 'mixed')
    num_questions = session.get('num_questions', 5)
    
    # Clear any previous quiz session completely
    session_keys = ['quiz_questions', 'current_question', 'quiz_score', 'user_answers', 'quiz_stored']
    for key in session_keys:
        session.pop(key, None)
    
    # Also reset the generator's current quiz tracking
    if hasattr(question_generator, 'current_quiz_questions'):
        question_generator.current_quiz_questions = set()
        question_generator.current_quiz_embeddings = []
    
    try:
        # Generate COMPLETELY FRESH questions based on previous settings
        if difficulty == 'mixed':
            questions = question_generator.generate_mixed_difficulty_quiz(category, num_questions)
            difficulty_display = 'Mixed (All Levels)'
        else:
            questions = question_generator.generate_dynamic_quiz_questions(difficulty, category, num_questions)
            difficulty_display = difficulty.title()
        
        if not questions:
            flash('Could not generate new questions. Please try again.', 'error')
            return redirect(url_for('index'))
        
        # Store new quiz session
        session['quiz_questions'] = questions
        session['current_question'] = 0
        session['quiz_score'] = 0
        session['quiz_category'] = category
        session['quiz_difficulty'] = difficulty
        session['num_questions'] = num_questions
        session['user_answers'] = {}
        session['quiz_stored'] = False
        
        print(f"🔄 NEW {difficulty_display} {category} quiz started with {len(questions)} COMPLETELY FRESH questions!")
        flash(f'New {difficulty_display} {category} quiz started with {len(questions)} COMPLETELY NEW questions!', 'success')
        return redirect(url_for('take_quiz'))
    except Exception as e:
        print(f"Error generating new quiz: {e}")
        flash(f'Error generating new quiz: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/restart-quiz-same-settings')
@login_required
def restart_quiz_same_settings():
    """Start a new quiz with the same settings but COMPLETELY FRESH questions"""
    # Get the previous quiz settings from session
    category = session.get('quiz_category', 'English Language')
    difficulty = session.get('quiz_difficulty', 'mixed')
    num_questions = session.get('num_questions', 5)
    
    # Clear previous quiz session completely
    session_keys = ['quiz_questions', 'current_question', 'quiz_score', 'user_answers', 'quiz_stored']
    for key in session_keys:
        session.pop(key, None)
    
    # Also reset the generator's current quiz tracking
    if hasattr(question_generator, 'current_quiz_questions'):
        question_generator.current_quiz_questions = set()
        question_generator.current_quiz_embeddings = []
    
    try:
        # Generate COMPLETELY FRESH questions based on previous settings
        if difficulty == 'mixed':
            questions = question_generator.generate_mixed_difficulty_quiz(category, num_questions)
            difficulty_display = 'Mixed (All Levels)'
        else:
            questions = question_generator.generate_dynamic_quiz_questions(difficulty, category, num_questions)
            difficulty_display = difficulty.title()
        
        if not questions:
            flash('Could not generate new questions. Please try again.', 'error')
            return redirect(url_for('index'))
        
        # Store new quiz session
        session['quiz_questions'] = questions
        session['current_question'] = 0
        session['quiz_score'] = 0
        session['quiz_category'] = category
        session['quiz_difficulty'] = difficulty
        session['num_questions'] = num_questions
        session['user_answers'] = {}
        session['quiz_stored'] = False
        
        print(f"🔄 NEW {difficulty_display} {category} quiz started with {len(questions)} COMPLETELY FRESH questions!")
        flash(f'New {difficulty_display} {category} quiz started with {len(questions)} COMPLETELY NEW questions!', 'success')
        return redirect(url_for('take_quiz'))
    except Exception as e:
        print(f"Error generating new quiz: {e}")
        flash(f'Error generating new quiz: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/quiz')
@login_required
def take_quiz():
    if 'quiz_questions' not in session:
        flash('No quiz in progress. Please start a new quiz.', 'error')
        return redirect(url_for('index'))
    
    questions = session.get('quiz_questions', [])
    current_idx = session.get('current_question', 0)
    
    if current_idx >= len(questions):
        return redirect(url_for('quiz_results'))
    
    current_question = questions[current_idx]
    return render_template_string(QUIZ_TEMPLATE, 
                         question=current_question,
                         question_number=current_idx + 1,
                         total_questions=len(questions))

@app.route('/submit-answer', methods=['POST'])
@login_required
def submit_answer():
    selected_answer = request.form.get('answer', '').strip()
    questions = session.get('quiz_questions', [])
    current_idx = session.get('current_question', 0)
    
    if 'user_answers' not in session:
        session['user_answers'] = {}
    
    session['user_answers'][str(current_idx)] = selected_answer
    
    if current_idx < len(questions):
        correct_answer = questions[current_idx].get('answer', '')
        
        # Use the improved answer checking
        if is_answer_correct(selected_answer, correct_answer):
            session['quiz_score'] = session.get('quiz_score', 0) + 1
            print(f"✅ Question {current_idx + 1} correct! User: '{selected_answer}' | Correct: '{correct_answer}'")
        else:
            print(f"❌ Question {current_idx + 1} incorrect. User: '{selected_answer}' | Correct: '{correct_answer}'")
    
    session['current_question'] = current_idx + 1
    
    if current_idx + 1 >= len(questions):
        return redirect(url_for('quiz_results'))
    
    return redirect(url_for('take_quiz'))

@app.route('/quiz-results')
@login_required
def quiz_results():
    # Get data from session
    questions = session.get('quiz_questions', [])
    score = session.get('quiz_score', 0)
    category = session.get('quiz_category', 'English Language')
    difficulty = session.get('quiz_difficulty', 'mixed')
    num_questions = session.get('num_questions', 5)
    user_answers = session.get('user_answers', {})
    
    total = len(questions)
    percentage = (score / total * 100) if total > 0 else 0
    
    # Store quiz results in database only if not already stored
    if questions and not session.get('quiz_stored', False):
        try:
            with sqlite3.connect(app.config['DATABASE']) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM categories WHERE name = ?', (category,))
                cat_result = cursor.fetchone()
                category_id = cat_result[0] if cat_result else 1
                
                cursor.execute('''
                    INSERT INTO quizzes (user_id, category_id, difficulty, score, total_questions)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session['user_id'], category_id, difficulty, score, len(questions)))
                conn.commit()
                session['quiz_stored'] = True
                print("Quiz results saved to database")
        except Exception as e:
            print(f"Error saving quiz results: {e}")
    
    # Debug: Print answer analysis
    print(f"\n=== QUIZ RESULTS ANALYSIS ===")
    print(f"Score: {score}/{total} ({percentage}%)")
    for i, question in enumerate(questions):
        user_answer = user_answers.get(str(i), "Not answered")
        correct_answer = question.get('answer', '')
        is_correct = is_answer_correct(user_answer, correct_answer)
        status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        print(f"Q{i+1}: {status}")
        print(f"  Question: {question['question']}")
        print(f"  User answer: '{user_answer}'")
        print(f"  Correct answer: '{correct_answer}'")
        print(f"  Normalized user: '{normalize_answer(user_answer)}'")
        print(f"  Normalized correct: '{normalize_answer(correct_answer)}'")
        print()
    
    # Render template with the data
    return render_template_string(QUIZ_RESULTS_TEMPLATE, 
                         questions=questions,
                         user_answers=user_answers,
                         score=score,
                         total=total,
                         percentage=int(round(percentage)))

@app.route('/clear-quiz-session')
@login_required
def clear_quiz_session():
    """Optional route to manually clear quiz session if needed"""
    session_keys = ['quiz_questions', 'current_question', 'quiz_score', 'quiz_category', 'quiz_difficulty', 'user_answers', 'quiz_stored', 'num_questions']
    for key in session_keys:
        session.pop(key, None)
    flash('Quiz session cleared', 'success')
    return redirect(url_for('index'))

@app.route('/reset-vector-db')
def reset_vector_db():
    """Reset the vector database"""
    if hasattr(question_generator, 'vector_db'):
        success = question_generator.vector_db.reset_database()
        if success:
            flash('Vector database reset successfully! New questions will be generated.', 'success')
        else:
            flash('Failed to reset vector database', 'error')
    else:
        flash('Vector database not available', 'error')
    return redirect(url_for('index'))

@app.route('/debug-questions')
def debug_questions():
    """Debug endpoint to see current question distribution"""
    if hasattr(question_generator, 'vector_db'):
        stats = {
            'total': question_generator.vector_db.get_question_count(),
            'easy': question_generator.vector_db.get_question_count('easy'),
            'medium': question_generator.vector_db.get_question_count('medium'), 
            'hard': question_generator.vector_db.get_question_count('hard'),
            'very_hard': question_generator.vector_db.get_question_count('very hard')
        }
        
        return jsonify({
            'stats': stats,
            'current_quiz_questions_count': len(question_generator.current_quiz_questions)
        })
    return jsonify({"error": "Vector DB not available"})

@app.route('/debug-vector-db')
def debug_vector_db():
    """Debug endpoint to see all questions in vector database"""
    if hasattr(question_generator, 'vector_db'):
        all_questions = question_generator.vector_db.get_all_questions()
        debug_info = {
            'total_questions': len(all_questions),
            'questions': all_questions[:10]  # First 10 questions
        }
        return jsonify(debug_info)
    return jsonify({"error": "Vector DB not available"})

@app.route('/force-reset-quiz')
@login_required
def force_reset_quiz():
    """Force reset the current quiz session"""
    session_keys = ['quiz_questions', 'current_question', 'quiz_score', 'quiz_category', 'quiz_difficulty', 'user_answers', 'quiz_stored', 'num_questions']
    for key in session_keys:
        session.pop(key, None)
    
    # Also reset current quiz questions in generator
    if hasattr(question_generator, 'current_quiz_questions'):
        question_generator.current_quiz_questions = set()
        question_generator.current_quiz_embeddings = []
    
    flash('Quiz session completely reset. You can start a fresh quiz now.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("\n" + "="*60)
    print("QuizMaster Pro - AI-Powered Fill-in-the-Blank Quizzes")
    print("MEANINGFUL QUESTIONS - No IDs, No Placeholders")
    print("="*60)
    print(f"Groq API Status: {('ACTIVE' if API_AVAILABLE else 'INACTIVE')}")
    
    if API_AVAILABLE and API_KEY:
        key_preview = API_KEY[:10] + "..." if len(API_KEY) > 10 else API_KEY
        print(f"API Key: {key_preview}")
        print("Questions are MEANINGFUL with NO IDs or placeholders")
        print("Every quiz generates brand new educational questions from LLM")
    else:
        print("WARNING: LLM is inactive - using meaningful fallback questions")
        print("To enable AI: Set GROQ_API_KEY in your environment variables")
        print("Get your key from: https://console.groq.com/keys")
    
    print(f"Database: {app.config['DATABASE']}")
    print(f"Vector DB: ./chroma_db/")
    print("Available Categories: English Language, Quantitative Aptitude, Reasoning Ability, General/Banking Awareness, Computer Aptitude")
    print("Debug Routes: /debug-vector-db, /force-reset-quiz")
    print("Server: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)