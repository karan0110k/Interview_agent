# agentic_interviewer_final_v2.py
import os
import uuid
import json
import logging
import subprocess
from typing import List, Dict, Any, Set, Optional, Tuple
from datetime import datetime
from enum import Enum

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import fitz  # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq, GroqError
from models import User, InterviewSession as DBInterviewSession

# ---- Configuration ----
class Config:
    """Centralized configuration for the application."""
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PORT = int(os.getenv("PORT", 8080))
    DEFAULT_QUESTIONS_PATH = "question_bank.json"
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required in your .env file.")

# ---- Setup Logging ----
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---- Constants ----
class Action(Enum):
    ASK_QUESTION = "ask_question"
    ASK_BEHAVIORAL = "ask_behavioral"
    GIVE_CODING = "give_coding"
    END_INTERVIEW = "end_interview"
    FALLBACK = "fallback"

# ---- LLM Client ----
class LLMClient:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def chat_completion(self, messages: List[Dict[str, str]], model: str = "llama3-8b-8192", temp: float = 0.6, max_tokens: int = 512) -> str:
        try:
            resp = self.client.chat.completions.create(messages=messages, model=model, temperature=temp, max_tokens=max_tokens)
            return resp.choices[0].message.content.strip()
        except GroqError as e:
            logger.error(f"Groq API error: {e}", exc_info=True)
            return "ERROR: The language model call failed."
        except Exception as e:
            logger.error(f"An unexpected error occurred during the LLM call: {e}", exc_info=True)
            return "ERROR: An unexpected error occurred."

    def chat_completion_json(self, messages: List[Dict[str, str]], retries: int = 2) -> Optional[Dict[str, Any]]:
        prompt_suffix = "\n\nReturn ONLY a valid JSON object and nothing else."
        messages[-1]["content"] += prompt_suffix
        for attempt in range(retries):
            try:
                response_text = self.chat_completion(messages, temp=0.2, max_tokens=1024)
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end != 0:
                    json_text = response_text[start:end]
                    return json.loads(json_text)
                else:
                    logger.warning(f"Could not find JSON in response (attempt {attempt+1}): {response_text}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON on attempt {attempt+1}. Response: {response_text}")
            except Exception as e:
                logger.error(f"Error during JSON completion: {e}", exc_info=True)
                return None
        return None

# ---- Question Bank & RAG System ----
class QuestionBank:
    def __init__(self, filepath: str):
        self.questions: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors: Optional[np.ndarray] = None
        self.docs: List[str] = []
        self._load_or_create(filepath)
        self._build_rag_index()

    def _load_or_create(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f: self.questions = json.load(f)
            logger.info(f"Loaded {len(self.questions)} questions from {filepath}.")
        else:
            self.questions = [
                {"id": "q1", "text": "Explain the difference between a process and a thread.", "tags": ["os", "concurrency"], "difficulty": "Easy"},
                {"id": "q2", "text": "How does a hash table handle collisions?", "tags": ["data-structures"], "difficulty": "Easy"},
                {"id": "q3", "text": "Design an O(n) algorithm to find the majority element in an array.", "tags": ["algorithms"], "difficulty": "Medium"},
            ]
            with open(filepath, "w", encoding="utf-8") as f: json.dump(self.questions, f, indent=2)
            logger.info(f"Created default question DB at {filepath}.")

    def _build_rag_index(self):
        self.docs = [q['text'] for q in self.questions]
        if not self.docs: return
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.doc_vectors = self.vectorizer.fit_transform(self.docs)
        logger.info(f"Built RAG index with {len(self.docs)} documents.")

    def find_relevant_question(self, query: str) -> Optional[str]:
        if not self.vectorizer or self.doc_vectors is None: return None
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = (self.doc_vectors * query_vector.T).toarray().flatten()
        if np.max(cosine_similarities) == 0: return None
        return self.docs[np.argmax(cosine_similarities)]

# ---- Interview Session Logic ----
class InterviewSession:
    def __init__(self, session_id: str, resume_text: str, llm_client: LLMClient, question_bank: QuestionBank, difficulty: str = "Medium"):
        self.id, self.llm, self.q_bank, self.difficulty = session_id, llm_client, question_bank, difficulty
        self.created_at = datetime.utcnow()
        self.resume_text, self.resume_summary, self.resume_keywords = resume_text, "", {}
        self.history: List[Dict[str, Any]] = []
        self.performance_score: float = 0.5
        self.asked_questions: Set[str] = set()

    def analyze_resume(self):
        self.resume_summary = self.llm.chat_completion([
            {"role": "system", "content": "You are an expert resume analyzer. Provide a comprehensive analysis (300-500 words) focusing on concrete skills, projects, and experiences."},
            {"role": "user", "content": f"Analyze this resume:\n\n{self.resume_text[:12000]}"}
        ], max_tokens=800)
        self.resume_keywords = self.llm.chat_completion_json([
            {"role": "system", "content": "Extract key information from a resume summary into JSON."},
            {"role": "user", "content": f"Extract technical skills, domain expertise, project types, and experience level from this summary:\n{self.resume_summary}"}
        ]) or {}

    def planner_decide(self) -> Dict[str, Any]:
        """Decides the next action, prioritizing direct user requests to change topics."""
        system = (
            "You are an expert, adaptive AI interview planner. You must adapt the interview based on the user's requests. "
            "Decide the next action. The difficulty MUST be exactly '{self.difficulty}'. "
            "Return a JSON object: {'action': 'action_name', 'difficulty': '{self.difficulty}', 'tags': ['relevant_tag']}"
        )
        history_text = "\n".join([f"{h['role'].title()}: {h.get('content', '')}" for h in self.history[-4:]]) # Focus on recent history

        user = f"""
        **CRITICAL INSTRUCTION:** Review the 'Recent History'. If the User's last message contains a direct request to change topics (e.g., "ask me about Java," "let's switch to SQL"), your next question MUST be on that new topic. Generate relevant 'tags' for it.

        Required difficulty: '{self.difficulty}'.
        Candidate Profile (use as a fallback if no topic change is requested): {self.resume_keywords}
        Recent History:
        {history_text}

        Questions Asked: {list(self.asked_questions)}

        Choose an action based on the CRITICAL INSTRUCTION first. If no instruction is given, proceed based on the profile.
        
        Example if user asked for Java:
        {{
          "action": "ask_question",
          "tags": ["java", "data-structures"],
          "difficulty": "{self.difficulty}"
        }}
        """
        plan = self.llm.chat_completion_json([{"role": "system", "content": system}, {"role": "user", "content": user}])
        
        if not plan or "action" not in plan:
            logger.warning("Planner failed. Using fallback logic.")
            action = Action.ASK_QUESTION if len(self.history) < 7 else Action.GIVE_CODING
            if len(self.history) > 10: action = Action.END_INTERVIEW
            return {"action": action.value, "difficulty": self.difficulty, "tags": []}
        
        plan['difficulty'] = self.difficulty
        if 'tags' not in plan:
            plan['tags'] = []
            
        return plan

    def _generate_question(self, action: Action, tags: List[str] = None, difficulty: str = None) -> str:
        """Generates a question, prioritizing a specified topic if provided."""
        max_attempts = 5
        difficulty_to_use = difficulty or self.difficulty
        system_prompt = "You are an expert interviewer. Your SOLE task is to generate a single question. Return ONLY the question text itself, without any preamble or quotation marks."

        for _ in range(max_attempts):
            q_text = ""
            if action in [Action.ASK_QUESTION, Action.GIVE_CODING]:
                query = self.resume_summary + " " + " ".join(tags or [])
                rag_q = self.q_bank.find_relevant_question(query)
                if rag_q and rag_q not in self.asked_questions:
                    q_text = rag_q
                else:
                    topic_context = f"on the topic of {', '.join(tags)}" if tags else ""
                    user_prompt = (f"Create ONE single, concise, {difficulty_to_use} "
                                   f"{'coding challenge' if action == Action.GIVE_CODING else 'technical question'} {topic_context}. "
                                   f"The candidate's general profile (for context only) is: {self.resume_keywords}. "
                                   f"Do NOT repeat these questions: {list(self.asked_questions)}")
                    q_text = self.llm.chat_completion([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], max_tokens=150)
            elif action == Action.ASK_BEHAVIORAL:
                user_prompt = (f"Create ONE single, concise behavioral question, under 25 words. "
                               f"Probe an experience from this summary: {self.resume_summary}. Don't repeat: {list(self.asked_questions)}")
                q_text = self.llm.chat_completion([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temp=0.7, max_tokens=120)
            
            if q_text and q_text not in self.asked_questions:
                self.asked_questions.add(q_text)
                return q_text
        return "Let's switch gears. Tell me about a time you had to learn a new technology quickly."

    def evaluate_answer(self, question: str, answer: str) -> Dict[str, Any]:
        system = "You are an objective interviewer. Score the answer from 0.0 to 1.0 and give brief, constructive feedback. Return JSON: {\"score\": float, \"feedback\": \"text\"}"
        user = f"Question: {question}\n\nAnswer:\n{answer}"
        evaluation = self.llm.chat_completion_json([{"role": "system", "content": system}, {"role": "user", "content": user}])
        if evaluation and isinstance(evaluation.get("score"), (int, float)):
            self.performance_score = (self.performance_score + evaluation["score"]) / 2
            evaluation["score"] = max(0.0, min(1.0, evaluation["score"]))
            return evaluation
        return {"score": 0.5, "feedback": "Could not automatically evaluate."}
        
    def generate_interview_summary(self) -> Dict[str, Any]:
        """Generates a comprehensive summary of the interview performance."""
        system = (
            "You are an expert interview assessor. Analyze the interview history and provide a detailed assessment. "
            "Return a JSON object with the following structure: "
            "{\"overall_rating\": string, \"strengths\": [string], \"areas_for_improvement\": [string], \"skill_assessment\": {skill: rating}, \"recommendations\": [string]}"
        )
        
        # Prepare history for analysis
        history_text = "\n".join([f"{h['role'].title()}: {h.get('content', '')}" for h in self.history])
        
        user = f"""
        Analyze this interview and provide a comprehensive assessment:
        
        Resume Summary: {self.resume_summary}
        
        Interview History:
        {history_text}
        
        Current Performance Score: {self.performance_score}
        
        Provide an honest, constructive assessment including:
        1. Overall rating (Excellent/Good/Average/Below Average/Poor)
        2. Key strengths demonstrated (3-5 points)
        3. Areas for improvement (3-5 points)
        4. Assessment of technical and soft skills mentioned in the interview
        5. Specific recommendations for improvement
        """
        
        summary = self.llm.chat_completion_json([{"role": "system", "content": system}, {"role": "user", "content": user}])
        
        if not summary:
            return {
                "overall_rating": "Assessment Unavailable",
                "strengths": ["Unable to generate strengths assessment"],
                "areas_for_improvement": ["Unable to generate improvement areas"],
                "skill_assessment": {},
                "recommendations": ["Please try again later"]
            }
            
        return summary

# ---- Flask App ----
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(user_id)

llm_client = LLMClient(api_key=Config.GROQ_API_KEY)
question_bank = QuestionBank(filepath=Config.DEFAULT_QUESTIONS_PATH)
SESSIONS: Dict[str, InterviewSession] = {}

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        
        user = User.get_by_email(email)
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate form data
        if not name or not email or not password or not confirm_password:
            flash('All fields are required', 'danger')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('signup.html')
        
        # Check if user already exists
        existing_user = User.get_by_email(email)
        if existing_user:
            flash('Email already registered', 'danger')
            return render_template('signup.html')
        
        # Create new user
        user = User(name=name, email=email)
        user.set_password(password)
        user.save()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's interview sessions
    sessions = DBInterviewSession.get_by_user_id(current_user.id)
    return render_template('dashboard.html', sessions=sessions)

@app.route('/session/<session_id>')
@login_required
def session_details(session_id):
    # Get the specific interview session
    try:
        session = DBInterviewSession.get_by_id(ObjectId(session_id))
        if not session or session.user_id != current_user.id:
            flash('Session not found', 'danger')
            return redirect(url_for('dashboard'))
        return render_template('session_details.html', session=session)
    except Exception as e:
        logger.error(f"Error in session_details: {e}", exc_info=True)
        flash('An error occurred while retrieving the session', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/update-profile', methods=['POST'])
@login_required
def update_profile():
    name = request.form.get('name')
    email = request.form.get('email')
    
    if not name or not email:
        flash('All fields are required', 'danger')
        return redirect(url_for('settings'))
    
    # Check if email is already taken by another user
    existing_user = User.get_by_email(email)
    if existing_user and existing_user.id != current_user.id:
        flash('Email already registered by another user', 'danger')
        return redirect(url_for('settings'))
    
    # Update user profile
    current_user.name = name
    current_user.email = email
    current_user.save()
    
    flash('Profile updated successfully', 'success')
    return redirect(url_for('settings'))

@app.route('/update-password', methods=['POST'])
@login_required
def update_password():
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    if not current_password or not new_password or not confirm_password:
        flash('All fields are required', 'danger')
        return redirect(url_for('settings'))
    
    if new_password != confirm_password:
        flash('New passwords do not match', 'danger')
        return redirect(url_for('settings'))
    
    if not current_user.check_password(current_password):
        flash('Current password is incorrect', 'danger')
        return redirect(url_for('settings'))
    
    # Update password
    current_user.set_password(new_password)
    current_user.save()
    
    flash('Password updated successfully', 'success')
    return redirect(url_for('settings'))

@app.route('/update-preferences', methods=['POST'])
@login_required
def update_preferences():
    default_difficulty = request.form.get('default_difficulty')
    focus_areas = request.form.getlist('focus_areas')
    
    # Here you would save these preferences to the user's profile
    # For now, we'll just show a success message
    flash('Interview preferences updated successfully', 'success')
    return redirect(url_for('settings'))

@app.route('/interview')
@login_required
def interview():
    return render_template('index.html')

@app.route('/start-interview', methods=['POST'])
@login_required
def start_interview():
    if 'resume' not in request.files: return jsonify({"error": "Resume file is required."}), 400
    try:
        resume_file, difficulty = request.files['resume'], request.form.get("difficulty", "Medium")
        with fitz.open(stream=resume_file.read(), filetype="pdf") as doc:
            resume_text = "".join(page.get_text() for page in doc)
        session_id = str(uuid.uuid4())
        session = InterviewSession(session_id, resume_text, llm_client, question_bank, difficulty)
        session.analyze_resume()
        
        welcome_message = "Welcome to InterviewProAI. To begin, please tell me a little about yourself and your background."
        session.history.append({"role": "assistant", "content": welcome_message, "meta": {"action": "introduction"}})
        SESSIONS[session_id] = session
        
        # Save to MongoDB
        db_session = DBInterviewSession(
            user_id=current_user.id,
            resume_text=resume_text,
            resume_summary=session.resume_summary,
            difficulty=difficulty,
            history=session.history,
            performance_score=session.performance_score
        )
        db_session.save()

        return jsonify({
            "session_id": session_id,
            "question": welcome_message,
            "resume_summary": session.resume_summary,
            "status": "awaiting_introduction"
        })
    except Exception as e:
        logger.error(f"Error in /start-interview: {e}", exc_info=True)
        return jsonify({"error": "Internal server error."}), 500

@app.route('/next-question', methods=['POST'])
@login_required
def next_question_route():
    try:
        data = request.get_json()
        session_id, answer = data.get('session_id'), data.get('answer', '')
        session = SESSIONS.get(session_id)
        if not session: return jsonify({"error": "Invalid session ID."}), 404
        
        # Handle the initial introduction from the user
        if len(session.history) == 1 and session.history[0]['meta']['action'] == 'introduction':
            session.history.append({"role": "user", "content": answer})
            first_action = Action.ASK_BEHAVIORAL
            first_question = session._generate_question(first_action, tags=session.resume_keywords.get('technical_skills'))
            session.history.append({"role": "assistant", "content": first_question, "meta": {"action": first_action.value}})
            
            # Update MongoDB session
            db_session = DBInterviewSession.get_by_user_id(current_user.id)[0]  # Get the most recent session
            db_session.history = session.history
            db_session.save()
            
            return jsonify({
                "question": first_question,
                "evaluation": {"score": None, "feedback": "Introduction received. Let's begin."},
                "performance": session.performance_score,
                "status": "in-progress"
            })
            
        last_question = next((h['content'] for h in reversed(session.history) if h['role'] == 'assistant'), "")
        session.history.append({"role": "user", "content": answer})
        
        eval_result = session.evaluate_answer(last_question, answer)
        session.history.append({"role": "system", "content": f"Evaluation: {eval_result}"})

        plan = session.planner_decide()
        action = Action(plan.get("action", Action.FALLBACK.value))

        if action == Action.END_INTERVIEW:
            final_message = "Thank you for your time. This concludes the interview."
            session.history.append({"role": "assistant", "content": final_message, "meta": {"action": action.value}})
            
            # Update MongoDB session
            db_session = DBInterviewSession.get_by_user_id(current_user.id)[0]  # Get the most recent session
            db_session.history = session.history
            db_session.performance_score = session.performance_score
            db_session.save()
            
            return jsonify({"question": final_message, "evaluation": eval_result, "performance": session.performance_score, "status": "finished"})

        next_q = session._generate_question(action, plan.get("tags"), plan.get("difficulty"))
        session.history.append({"role": "assistant", "content": next_q, "meta": {"action": action.value}})

        # Update MongoDB session
        db_session = DBInterviewSession.get_by_user_id(current_user.id)[0]  # Get the most recent session
        db_session.history = session.history
        db_session.performance_score = session.performance_score
        db_session.save()

        return jsonify({"question": next_q, "evaluation": eval_result, "performance": session.performance_score, "status": "in-progress"})
    except Exception as e:
        logger.error(f"Error in /next-question: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate next question."}), 500

@app.route('/interview-summary', methods=['POST'])
@login_required
def interview_summary():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        session = SESSIONS.get(session_id)
        
        if not session:
            return jsonify({"error": "Invalid session ID."}), 404
            
        summary = session.generate_interview_summary()
        
        # Calculate detailed performance metrics
        technical_score = 0
        communication_score = 0
        problem_solving_score = 0
        relevant_experience_score = 0
        
        question_count = 0
        technical_count = 0
        communication_count = 0
        problem_solving_count = 0
        relevant_experience_count = 0
        
        # Analyze history to categorize questions and calculate specific scores
        for entry in session.history:
            if entry.get('role') == 'system' and 'Evaluation' in entry.get('content', ''):
                eval_text = entry.get('content', '')
                try:
                    # Extract score from evaluation
                    score_str = eval_text.split('score": ')[1].split(',')[0]
                    score = float(score_str)
                    
                    # Find the corresponding question
                    idx = session.history.index(entry)
                    if idx >= 2:  # Make sure we have a question before the answer
                        question_text = session.history[idx-2].get('content', '').lower()
                        
                        question_count += 1
                        
                        # Categorize questions
                        if any(keyword in question_text for keyword in ['technical', 'technology', 'programming', 'code', 'algorithm']):
                            technical_score += score
                            technical_count += 1
                        
                        if any(keyword in question_text for keyword in ['communicate', 'explain', 'team', 'collaboration']):
                            communication_score += score
                            communication_count += 1
                        
                        if any(keyword in question_text for keyword in ['problem', 'solve', 'challenge', 'approach']):
                            problem_solving_score += score
                            problem_solving_count += 1
                        
                        if any(keyword in question_text for keyword in ['experience', 'project', 'work', 'achievement']):
                            relevant_experience_score += score
                            relevant_experience_count += 1
                except:
                    pass
        
        # Normalize scores
        technical_score = (technical_score / technical_count) if technical_count > 0 else 0
        communication_score = (communication_score / communication_count) if communication_count > 0 else 0
        problem_solving_score = (problem_solving_score / problem_solving_count) if problem_solving_count > 0 else 0
        relevant_experience_score = (relevant_experience_score / relevant_experience_count) if relevant_experience_count > 0 else 0
        
        # Create detailed performance metrics
        performance_metrics = {
            'technical_score': technical_score,
            'communication_score': communication_score,
            'problem_solving_score': problem_solving_score,
            'relevant_experience_score': relevant_experience_score,
            'question_count': question_count
        }
        
        # Add performance metrics to summary
        if isinstance(summary, dict):
            summary['performance_metrics'] = performance_metrics
        
        # Update MongoDB session with final summary
        db_session = DBInterviewSession.get_by_user_id(current_user.id)[0]  # Get the most recent session
        db_session.summary = summary
        db_session.performance_score = session.performance_score
        db_session.performance_metrics = performance_metrics
        db_session.save()
        
        return jsonify({
            "summary": summary,
            "performance_score": session.performance_score,
            "performance_metrics": performance_metrics,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error in /interview-summary: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate interview summary."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Config.PORT, debug=True)