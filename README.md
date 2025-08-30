# InterviewPro AI

An agentic AI-powered interview preparation platform that helps users practice and improve their interview skills through simulated interviews with personalized feedback.

## Features

- User authentication system with login/signup functionality
- Personalized dashboard to track interview history and performance
- AI-powered interview simulation with real-time feedback
- Resume analysis to tailor interview questions to your experience
- Comprehensive interview summary with performance metrics
- MongoDB integration for secure data storage

## Prerequisites

- Python 3.8+
- MongoDB running locally on port 27017
- Groq API key for LLM access

## Installation

1. Clone the repository

```bash
git clone <repository-url>
cd Job_interview
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set up environment variables

Create a `.env` file in the root directory with the following variables:

```
GROQ_API_KEY=your_groq_api_key
FLASK_SECRET_KEY=your_secret_key
```

4. Initialize the database

```bash
python init_db.py
```

This will create the necessary collections in MongoDB and a default admin user (email: admin@example.com, password: admin123).

## Running the Application

```bash
python app.py
```

The application will be available at http://localhost:5000

## Usage

1. Register a new account or log in with existing credentials
2. Navigate to the dashboard to view past interviews or start a new one
3. For a new interview, upload your resume and select the difficulty level
4. Answer the AI interviewer's questions using text or voice input
5. Receive real-time feedback on your answers
6. View a comprehensive summary at the end of the interview

## Project Structure

- `app.py`: Main application file with Flask routes and core functionality
- `models.py`: MongoDB models for user and interview session data
- `templates/`: HTML templates for the web interface
- `static/`: CSS, JavaScript, and other static assets
- `init_db.py`: Script to initialize the MongoDB database

## License

MIT