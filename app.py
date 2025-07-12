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
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=['*'])  # Enable CORS for Flutter app integration

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
        
        # Enhanced prompt for better civic issue detection
        prompt = """
        You are an AI assistant specialized in detecting civic infrastructure issues. 
        Analyze this image carefully and look for these specific civic problems:

        1. ROAD ISSUES: Potholes, cracks, damaged asphalt, road surface problems
        2. WASTE MANAGEMENT: Garbage accumulation, overflowing bins, illegal dumping, littering
        3. WATER/SEWAGE: Sewage overflows, drainage blockages, water leakage, flooding
        4. INFRASTRUCTURE: Broken streetlights, damaged sidewalks, faulty traffic signals
        5. PUBLIC PROPERTY: Damaged benches, broken signage, vandalism

        If you detect ANY civic issues, respond with this EXACT JSON format:
        {
            "issues_found": true,
            "issues": [
                {
                    "category": "one of: pothole, garbage, sewage, infrastructure, drainage, streetlight, sidewalk, traffic_signal, illegal_dumping, waterlogging",
                    "description": "Clear, detailed description of what you see (2-3 sentences)",
                    "severity": "low, medium, or high"
                }
            ]
        }

        If NO civic issues are detected, respond with:
        {
            "issues_found": false,
            "message": "No civic issues detected in the image"
        }

        IMPORTANT: 
        - Only respond with valid JSON
        - Be specific about what you see
        - Focus only on civic infrastructure problems
        - Ignore people, vehicles, or private property unless they're causing civic issues
        """
        
        # Generate response from Gemini
        response = model.generate_content([prompt, image])
        response_text = response.text.strip()
        
        logger.info(f"Gemini raw response: {response_text}")
        
        # Clean and parse JSON response
        try:
            # Remove any markdown formatting
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            # Find JSON content
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                
                result = json.loads(json_str)
                
                # Validate response structure
                if 'issues_found' not in result:
                    raise ValueError("Invalid response format")
                
                return result
            else:
                # Fallback: create response based on keywords
                civic_keywords = ['pothole', 'garbage', 'waste', 'sewage', 'drain', 
                                'street', 'light', 'road', 'damage', 'broken', 'overflow']
                
                if any(keyword in response_text.lower() for keyword in civic_keywords):
                    return {
                        "issues_found": True,
                        "issues": [
                            {
                                "category": "general",
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
                    
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text: {response_text}")
            
            # Fallback response
            return {
                "issues_found": False,
                "message": "No civic issues detected in the image",
                "debug_info": f"JSON parse error: {str(e)}"
            }
                
    except Exception as e:
        logger.error(f"Error processing image with Gemini: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test Gemini connection
        test_response = model.generate_content("Hello")
        gemini_status = True if test_response else False
    except:
        gemini_status = False
    
    return jsonify({
        "status": "healthy",
        "message": "Civic Issue Detection API is running",
        "gemini_available": gemini_status,
        "timestamp": int(time.time())
    }), 200

@app.route('/report_civic_issue', methods=['POST'])
def report_civic_issue():
    """Analyze image for civic issues"""
    
    # Log request details
    logger.info(f"Received POST request to /report_civic_issue")
    logger.info(f"Request headers: {dict(request.headers)}")
    logger.info(f"Request files: {list(request.files.keys())}")
    logger.info(f"Content type: {request.content_type}")
    
    try:
        # Check if request has file
        if 'image' not in request.files:
            logger.error("No image file provided in request")
            return jsonify({
                "success": False,
                "error": "No image file provided",
                "message": "Please upload an image file"
            }), 400
        
        file = request.files['image']
        logger.info(f"Received file: {file.filename}, Content type: {file.content_type}")
        
        # Check if file is selected
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({
                "success": False,
                "error": "No file selected",
                "message": "Please select an image file"
            }), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({
                "success": False,
                "error": "Invalid file type",
                "message": f"Allowed file types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Read image data
        image_data = file.read()
        logger.info(f"Image data size: {len(image_data)} bytes")
        
        # Validate image size
        if len(image_data) == 0:
            logger.error("Empty file received")
            return jsonify({
                "success": False,
                "error": "Empty file",
                "message": "The uploaded file is empty"
            }), 400
        
        # Validate image content
        try:
            test_image = Image.open(io.BytesIO(image_data))
            logger.info(f"Image format: {test_image.format}, Size: {test_image.size}, Mode: {test_image.mode}")
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            return jsonify({
                "success": False,
                "error": "Invalid image file",
                "message": "The uploaded file is not a valid image"
            }), 400
        
        # Process image with Gemini
        logger.info("Processing image with Gemini...")
        result = process_image_with_gemini(image_data)
        logger.info(f"Gemini analysis result: {result}")
        
        # Format response for Flutter app
        if result.get('issues_found', False):
            response_data = {
                "success": True,
                "issues_detected": True,
                "issues": result.get('issues', []),
                "count": len(result.get('issues', [])),
                "timestamp": int(time.time()),
                "message": f"Found {len(result.get('issues', []))} civic issue(s)"
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
        
        logger.info(f"Sending response: {response_data}")
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in report_civic_issue: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "message": "An error occurred while processing the image",
            "debug_info": str(e) if app.debug else None
        }), 500

@app.route('/test_gemini', methods=['GET'])
def test_gemini():
    """Test Gemini API connection"""
    try:
        # Test with a simple text prompt
        response = model.generate_content("Respond with 'Hello from Gemini!' if you are working correctly.")
        return jsonify({
            "status": "success",
            "message": "Gemini API is working",
            "response": response.text,
            "timestamp": int(time.time())
        }), 200
    except Exception as e:
        logger.error(f"Gemini API test failed: {e}")
        return jsonify({
            "status": "error",
            "message": f"Gemini API error: {str(e)}"
        }), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get available issue categories"""
    categories = [
        {"id": "pothole", "name": "Pothole", "description": "Road surface damage and potholes"},
        {"id": "garbage", "name": "Garbage/Waste", "description": "Waste management issues"},
        {"id": "sewage", "name": "Sewage", "description": "Sewage overflow and drainage problems"},
        {"id": "streetlight", "name": "Street Light", "description": "Street lighting issues"},
        {"id": "infrastructure", "name": "Infrastructure", "description": "General infrastructure damage"},
        {"id": "drainage", "name": "Drainage", "description": "Blocked drains and waterlogging"},
        {"id": "sidewalk", "name": "Sidewalk", "description": "Damaged sidewalks and footpaths"},
        {"id": "traffic_signal", "name": "Traffic Signal", "description": "Traffic signal problems"},
        {"id": "illegal_dumping", "name": "Illegal Dumping", "description": "Unauthorized waste disposal"},
        {"id": "waterlogging", "name": "Waterlogging", "description": "Water accumulation issues"}
    ]
    
    return jsonify({
        "success": True,
        "categories": categories,
        "count": len(categories)
    }), 200

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        "success": False,
        "error": "File too large",
        "message": "Maximum file size is 16MB"
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "message": "Something went wrong on our end"
    }), 500

@app.errorhandler(404)
def not_found(e):
    """Handle not found errors"""
    return jsonify({
        "success": False,
        "error": "Not found",
        "message": "The requested endpoint was not found"
    }), 404

# Add request logging middleware
@app.before_request
def log_request_info():
    logger.info(f"Request: {request.method} {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")

@app.after_request
def log_response_info(response):
    logger.info(f"Response: {response.status_code}")
    return response

if __name__ == '__main__':
    # Check if Gemini API key is set
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY environment variable is not set!")
        print("Please set it with: export GEMINI_API_KEY=your_api_key_here")
        exit(1)
    
    print("=" * 60)
    print("🚀 Starting Civic Issue Detection API...")
    print("=" * 60)
    print("Available endpoints:")
    print("  POST /report_civic_issue - Analyze image for civic issues")
    print("  GET /health - Health check")
    print("  GET /test_gemini - Test Gemini API connection")
    print("  GET /categories - Get available issue categories")
    print("=" * 60)
    
    # Test Gemini connection on startup
    try:
        test_response = model.generate_content("Test connection")
        print("✅ Gemini API connection successful")
    except Exception as e:
        print(f"❌ Gemini API connection failed: {e}")
        print("⚠️  The API will start but image analysis may not work")
    
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)