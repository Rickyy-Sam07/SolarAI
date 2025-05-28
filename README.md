AI Solar Rooftop Analyzer

An advanced web application that uses artificial intelligence and computer vision to analyze rooftops from satellite imagery, providing comprehensive solar installation assessments including system sizing, financial analysis, and environmental impact calculations.

🚀 Live Demo
Try it now: https://web-production-7ed9d.up.railway.app/

✨ Features
🔍 AI-Powered Analysis
Computer Vision Processing: Advanced OpenCV algorithms for roof detection

Obstacle Identification: Automatic detection of chimneys, vents, HVAC units

Shading Analysis: Comprehensive shadow pattern evaluation

Roof Orientation Detection: Precise cardinal direction identification

📊 Comprehensive Assessment
System Sizing: Optimal solar panel count and capacity calculations

Energy Production: Annual and monthly energy generation estimates

Financial Analysis: ROI, payback period, and lifetime savings

Environmental Impact: CO₂ offset and sustainability metrics

🏠 Property Analysis
Roof Area Calculation: Precise usable area determination

Structural Assessment: Roof condition and suitability evaluation

Multiple Panel Types: Support for various solar panel technologies

Location-Specific Data: Regional solar irradiance and weather factors

🛠️ Technology Stack
Backend
Flask 3.0.0 - Web framework

OpenCV 4.8.1 - Computer vision processing

NumPy 1.24.3 - Numerical computing

Pillow 10.0.1 - Image processing

Gunicorn 21.2.0 - Production server

Frontend
HTML5/CSS3 - Modern web standards

JavaScript ES6+ - Interactive functionality

Responsive Design - Mobile-first approach

Deployment
Railway.app - Cloud platform

Docker - Containerization

GitHub Actions - CI/CD pipeline🚀 Quick Start
Prerequisites
Python 3.11+

pip package manager

Installation
Clone the repository

bash
git clone https://github.com/Rickyy-Sam07/SolarAI.git
cd SolarAI
Create virtual environment

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Set environment variables

bash
export FLASK_ENV=development
export PORT=5000
Run the application

bash
python app.py
Open in browser

text
http://localhost:5000
📖 Usage Guide
1. Image Input
URL Method: Enter direct link to satellite image

Upload Method: Select local image file (JPG, PNG, WebP up to 10MB)

2. Property Details
Address: Property location for regional solar data

Electricity Rate: Current rate per kWh

Panel Type: Choose from available technologies

Manual Override: Optional roof area input

3. Analysis Results
Roof Analysis: Area, orientation, shading, condition

System Design: Capacity, panel count, annual production

Financial Analysis: Costs, savings, payback period, ROI

Environmental Impact: CO₂ offset, sustainability metrics

🔧 API Documentation
Base URL
text
(https://web-production-7ed9d.up.railway.app/)
Endpoints
POST /analyze-solar
Performs comprehensive solar analysis.

Request:

json
{
  "imageData": {
    "type": "url",
    "data": "https://example.com/satellite-image.jpg"
  },
  "address": "karnataka, india,
  "electricityRate": 0.12,
  "panelType": "monocrystalline",
  "roofArea": 2000
}
Response:

json
{
  "roofAnalysis": {
    "detectedArea": 1800,
    "usableArea": 1440,
    "orientation": "south",
    "shading": 12,
    "condition": "good"
  },
  "systemSpecs": {
    "capacity": 8.64,
    "panelCount": 18,
    "annualProduction": 12960
  },
  "financial": {
    "grossCost": 24192,
    "netCost": 16934,
    "paybackPeriod": 10.9,
    "roiPercentage": 156
  },
  "environmental": {
    "annualCo2Offset": 5.18,
    "treesEquivalent": 238
  },
  "confidence": 88,
  "analysisMethod": "Computer Vision Analysis"
}
GET /health
Returns application health status.

🚀 Deployment
Railway.app (Recommended)
Prepare for deployment

bash
# Ensure these files exist:
# - requirements.txt
# - Procfile
# - app.py
Deploy to Railway

bash
# Push to GitHub
git add .
git commit -m "Deploy to Railway"
git push origin main

# Connect to Railway
# 1. Go to railway.app
# 2. Connect GitHub repository
# 3. Deploy automatically
Set environment variables

OPENROUTER_API_KEY: Your AI service API key

PORT: Application port (auto-set by Railway)

Alternative Platforms
Render.com: Similar GitHub integration

Heroku: Traditional PaaS platform

DigitalOcean App Platform: Scalable deployment

🧪 Testing
Manual Testing
bash
# Test health endpoint
curl https://web-production-7ed9d.up.railway.app/api/health

# Test analysis endpoint
curl -X POST https://web-production-7ed9d.up.railway.app/api/analyze-solar \
  -H "Content-Type: application/json" \
  -d '{"imageData":{"type":"url","data":"IMAGE_URL"},"address":"Austin, TX","electricityRate":0.12,"panelType":"monocrystalline"}'
Performance Metrics
Analysis Speed: 45-60 seconds average

Accuracy: 85%+ confidence scores

Uptime: 99.5% availability target

🤝 Contributing
We welcome contributions! Please follow these steps:

Fork the repository

Create feature branch

bash
git checkout -b feature/amazing-feature
Commit changes

bash
git commit -m "Add amazing feature"
Push to branch

bash
git push origin feature/amazing-feature
Open Pull Request

Development Guidelines
Follow PEP 8 style guide

Add tests for new features

Update documentation

Ensure all tests pass

📊 Performance
System Requirements
Memory: 512MB minimum

CPU: 1 vCPU recommended

Storage: 1GB for dependencies

Optimization Features
Headless OpenCV: Server-optimized computer vision

Efficient Algorithms: Streamlined processing pipeline

Error Handling: Comprehensive fallback mechanisms

Caching: Regional data optimization

🔮 Roadmap
Q2 2025
 Enhanced computer vision models

 Improved accuracy for complex roof shapes

 Mobile app development

Q3 2025
 Real-time satellite data integration

 Advanced shading analysis with LIDAR

 Multi-language support

Q4 2025
 Commercial building support

 API authentication and rate limiting

 Advanced analytics dashboard

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
OpenCV Community - Computer vision algorithms

Flask Team - Web framework

Railway.app - Deployment platform

Solar Industry - Domain expertise and validation

📞 Support
Issues and Bugs
GitHub Issues: Report bugs and request features

Documentation: Comprehensive guides and API docs

Contact
Email: sambhranta1123@gmailcom

LinkedIn: www.linkedin.com/in/sambhranta-ghosh-995718277



⭐ Star this repository if you find it helpful!

🔗 Live Demo: https://web-production-7ed9d.up.railway.app/

Built with ❤️ for a sustainable future
