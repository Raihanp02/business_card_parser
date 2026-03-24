# Business Card Parser

A machine learning-powered business card parser that extracts structured information from business card images using OCR and Named Entity Recognition (NER).

## Overview

This project combines optical character recognition (OCR) with natural language processing (NLP) to automatically extract and categorize business card information including:

- **Person**: Contact name
- **Company**: Company name
- **Phone**: Phone numbers
- **Email**: Email addresses
- **Position**: Job title/position
- **Link**: Website/social links
- **Address**: Physical address

## Architecture

```
business_card_parser/
├── app/                          # Main application code
│   ├── api.py                   # FastAPI endpoints
│   ├── ai_services/             # AI/ML services
│   │   ├── ner.py               # Named Entity Recognition
│   │   └── ocr/
│   │       ├── ocr.py           # PaddleOCR wrapper
│   │       └── postprocess.py   # OCR post-processing
│   └── pipelines/
│       └── ocr_parser.py        # Main processing pipeline
├── assets/
│   └── models/                  # Pre-trained models
│       ├── whl/                 # PaddleOCR models
│       └── xlmr_tokenizer/      # XLM-RoBERTa tokenizer
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose setup
└── requirements.txt             # Python dependencies
```

## Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose (optional)
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd business_card_parser
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download/prepare models:
   - PaddleOCR models are downloaded automatically on first run
   - Download XLM-RoBERTa pre-trained models: https://drive.google.com/uc?id=160vL5_jhXjJK04J-EQ5ahw10zCJ_ni8t and placed it in `assets/models/`
   

### Local Development

1. Start the API server:
```bash
uvicorn app.api:app --reload --port 8000
```

2. Test the API:
```bash
curl -X POST "http://localhost:8000/parse-card" \
  -H "accept: application/json" \
  -F "file=@business_card.jpg"
```

### Docker Deployment

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /health
```
Returns: `{"status": "ok"}`

### Parse Business Card
```
POST /parse-card
```
**Parameters:**
- `file` (required): Image file (JPEG, PNG, etc.)

**Response:**
```json
{
  "result": {
    "Person": ["John Doe"],
    "Company": ["Tech Corp"],
    "Phone": ["+1-234-567-8900"],
    "Email": ["john.doe@techcorp.com"],
    "Position": ["Senior Developer"],
    "Link": ["www.techcorp.com"],
    "Address": ["123 Tech Street, San Francisco, CA"]
  }
}
```

## Components

### OCR Engine (PaddleOCR)
- Handles image text extraction
- Supports multiple languages (Latin, Chinese, etc.)
- Custom model paths for optimization
- See [OCR README](./app/ai_services/ocr/README.md)

### NER Service (XLM-RoBERTa)
- Extracts entities from OCR text
- 14 entity tags (BIO format)
- GPU/CPU compatible
- See [NER README](./app/ai_services/README.md)

### Processing Pipeline
- Orchestrates OCR → NER workflow
- Combines text extraction with entity recognition
- See [Pipeline README](./app/pipelines/README.md)

## Models

The project uses pre-trained models:

- **PaddleOCR v3/v4**: Recognition and detection models in `assets/models/whl/`
- **XLM-RoBERTa**: Multilingual BERT for NER in `assets/models/xlmr_tokenizer/`

Model download and caching is handled automatically.

## Dependencies

Key libraries:
- **FastAPI**: Web framework
- **PaddleOCR**: Optical character recognition
- **PyTorch**: Deep learning framework
- **Hugging Face Tokenizers**: Tokenization
- **Pillow & OpenCV**: Image processing

See [requirements.txt](./requirements.txt) for full list.

## Configuration

### Environment Variables
- `FLAGS_use_mkldnn`: Set to "0" to disable MKL-DNN (for compatibility)

### Performance Tuning
- GPU support enabled via CUDA (auto-detected)
- Models can be customized in initialization
- See individual component docs for tuning options

## Development

### Project Structure
- `app/`: Main application code
- `assets/`: Models and resources
- `Dockerfile`: Container configuration
- `docker-compose.yml`: Multi-container orchestration

### Testing
```bash
# Test specific endpoint
curl -X POST "http://localhost:8000/parse-card" \
  -F "file=@test_image.jpg"

# Check server health
curl "http://localhost:8000/health"
```

## Troubleshooting

- **Model Loading Issues**: Ensure `assets/models/` directory exists and models are properly placed
- **GPU Memory**: Set `use_gpu=False` in OCR config if running into memory issues
- **Image Format**: Supported formats: JPEG, PNG, BMP, TIFF

## License

[Specify your license here]

## Contributing

[Contribution guidelines]

## Support

For issues and questions, please create an issue in the repository.
