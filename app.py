# Simple Meeting AI - KIU Consulting
from flask import Flask, request, render_template, jsonify
import openai
import os
import json
import sqlite3

from werkzeug.utils import secure_filename

# Load environment variables
def load_env():
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

load_env()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Create directories
os.makedirs('uploads', exist_ok=True)

# Initialize OpenAI with error handling
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("ERROR: OPENAI_API_KEY not found!")
    print("Create a .env file with: OPENAI_API_KEY=your-key-here")
    exit(1)

try:
    client = openai.OpenAI(api_key=api_key)
    # Test the client
    print("✅ OpenAI client initialized successfully")
except Exception as e:
    print(f"❌ Error initializing OpenAI client: {e}")
    print("Try: pip install --upgrade openai")
    exit(1)

# Initialize database
def init_db():
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            summary TEXT,
            action_items TEXT,
            transcript TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# AI Processing
def process_meeting(audio_path):
    # 1. Transcribe with Whisper
    with open(audio_path, "rb") as audio:
        transcript_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio
        )
    
    # Extract text from transcript response
    if hasattr(transcript_response, 'text'):
        transcript_text = transcript_response.text
    else:
        transcript_text = str(transcript_response)
    
    print(f"Transcript: {transcript_text[:200]}...")  # Debug print
    
    # 2. Analyze with GPT-4
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system", 
                "content": "You are a meeting assistant. Extract a summary and action items from the transcript. Return your response as JSON with 'summary' and 'action_items' fields. Make the summary 1-2 sentences. Action items should be an array of strings."
            },
            {
                "role": "user", 
                "content": f"Please analyze this meeting transcript:\n\n{transcript_text}"
            }
        ],
        temperature=0.3
    )
    
    try:
        # Try to parse as JSON
        analysis_text = response.choices[0].message.content
        print(f"GPT-4 Response: {analysis_text}")  # Debug print
        
        analysis = json.loads(analysis_text)
    except json.JSONDecodeError:
        # If not valid JSON, create a simple structure
        analysis_text = response.choices[0].message.content
        analysis = {
            "summary": analysis_text[:300] + "..." if len(analysis_text) > 300 else analysis_text,
            "action_items": ["Review meeting notes", "Follow up on discussed items"]
        }
    
    return transcript_text, analysis

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    file.save(filepath)
    
    try:
        # Process with AI
        transcript, analysis = process_meeting(filepath)
        
        # Save to database
        conn = sqlite3.connect('meetings.db')
        cursor = conn.cursor()
        cursor.execute('''
    INSERT INTO meetings (filename, summary, action_items, transcript)
    VALUES (?, ?, ?, ?)
''', (filename, analysis['summary'], json.dumps(analysis['action_items']), transcript))

        conn.commit()
        conn.close()
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'summary': analysis['summary'],
            'action_items': analysis['action_items'],
            'transcript': transcript[:500] + "..." if len(transcript) > 500 else transcript

        })
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/meetings')
def get_meetings():
    conn = sqlite3.connect('meetings.db')
    cursor = conn.cursor()
    cursor.execute('SELECT filename, summary, action_items, created_at FROM meetings ORDER BY created_at DESC')
    meetings = []
    for row in cursor.fetchall():
        meetings.append({
            'filename': row[0],
            'summary': row[1],
            'action_items': json.loads(row[2]) if row[2] else [],
            'date': row[3]
        })
    conn.close()
    return jsonify(meetings)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)