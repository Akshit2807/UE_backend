import os
import io
import base64
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import google.generativeai as genai
from flask_cors import CORS
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app integration

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

from dotenv import load_dotenv
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_with_gemini(image_data):
    """Process image with Gemini API to detect civic issues"""
    try:
        # Create PIL Image from bytes
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Prepare prompt for civic issue detection
        prompt = """
        Analyze this image for civic issues and infrastructure problems. Look for:
        1. Potholes or road damage
        2. Garbage/waste management issues
        3. Sewage overflows or drainage problems
        4. Broken streetlights or infrastructure
        5. Damaged public property
        6. Illegal dumping
        7. Blocked drains or waterlogging
        8. Damaged sidewalks or footpaths
        9. Traffic signal issues
        10. Public facility damage

        If you find any civic issues, respond with a JSON format like this:
        {
            "issues_found": true,
            "issues": [
                {
                    "category": "pothole",
                    "description": "Large pothole on road surface causing traffic disruption",
                    "severity": "high"
                }
            ]
        }

        If no civic issues are detected, respond with:
        {
            "issues_found": false,
            "message": "No civic issues detected in the image"
        }

        Severity levels: "low", "medium", "high"
        Common issue types: "pothole", "garbage", "sewage", "drainage", "streetlight", "infrastructure", "illegal_dumping", "waterlogging", "sidewalk", "traffic_signal"
        """
        
        # Generate response from Gemini
        response = model.generate_content([prompt, image])
        
        # Parse the response
        response_text = response.text.strip()
        
        # Try to extract JSON from response
        import json
        try:
            # Sometimes the response might have additional text, so we need to find the JSON part
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                return result
            else:
                # If no JSON found, create a default response
                return {
                    "issues_found": False,
                    "message": "No civic issues detected in the image"
                }
        except json.JSONDecodeError:
            # If JSON parsing fails, create a response based on text content
            if any(keyword in response_text.lower() for keyword in ['pothole', 'garbage', 'sewage', 'damage', 'issue']):
                return {
                    "issues_found": True,
                    "issues": [
                        {
                            "type": "general",
                            "description": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                            "severity": "medium"
                        }
                    ]
                }
            else:
                return {
                    "issues_found": False,
                    "message": "No civic issues detected in the image"
                }
                
    except Exception as e:
        logger.error(f"Error processing image with Gemini: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Civic Issue Detection API is running"
    }), 200

@app.route('/report_civic_issue', methods=['POST'])
def report_civic_issue():
    """Analyze image for civic issues"""
    try:
        # Check if request has file
        if 'image' not in request.files:
            return jsonify({
                "error": "No image file provided",
                "message": "Please upload an image file"
            }), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                "error": "No file selected",
                "message": "Please select an image file"
            }), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({
                "error": "Invalid file type",
                "message": f"Allowed file types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Read image data
        image_data = file.read()
        
        # Validate image size
        if len(image_data) == 0:
            return jsonify({
                "error": "Empty file",
                "message": "The uploaded file is empty"
            }), 400
        
        # Process image with Gemini
        result = process_image_with_gemini(image_data)
        
        # Format response for Flutter app
        if result.get('issues_found', False):
            response_data = {
                "success": True,
                "issues_detected": True,
                "issues": result.get('issues', []),
                "count": len(result.get('issues', [])),
                "timestamp": int(time.time())
            }
        else:
            response_data = {
                "success": True,
                "issues_detected": False,
                "message": result.get('message', 'No civic issues detected in the image'),
                "issues": [],
                "count": 0,
                "timestamp": int(time.time())
            }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in report_civic_issue: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "message": "An error occurred while processing the image"
        }), 500

@app.route('/test_gemini', methods=['GET'])
def test_gemini():
    """Test Gemini API connection"""
    try:
        # Test with a simple text prompt
        response = model.generate_content("Hello, are you working?")
        return jsonify({
            "status": "success",
            "message": "Gemini API is working",
            "response": response.text
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Gemini API error: {str(e)}"
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        "error": "File too large",
        "message": "Maximum file size is 16MB"
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong on our end"
    }), 500

if __name__ == '__main__':
    # Check if Gemini API key is set
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY environment variable is not set!")
        print("Please set it with: export GEMINI_API_KEY=your_api_key_here")
        exit(1)
    
    print("Starting Civic Issue Detection API...")
    print("Available endpoints:")
    print("  POST /report_civic_issue - Analyze image for civic issues")
    print("  GET /health - Health check")
    print("  GET /test_gemini - Test Gemini API connection")
    
    app.run(debug=True, host='0.0.0.0', port=5000)