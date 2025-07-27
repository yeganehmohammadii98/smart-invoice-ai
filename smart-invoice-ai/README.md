# Smart Invoice AI System

An intelligent invoice processing system that uses machine learning to extract key information from invoice documents and improves over time through user feedback.

## Features

- Upload multiple invoice formats (PDF, images, CSV)
- Automatic field extraction using OCR and ML
- User correction tracking for continuous improvement
- Auto fine-tuning capabilities
- Clean web interface built with Streamlit

## Setup

1. Clone this repository
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the application: `streamlit run app.py`

## Project Status

- ✅ Phase 1: Basic structure and UI (Current)
- ⏳ Phase 2: OCR and document processing
- ⏳ Phase 3: Field extraction
- ⏳ Phase 4: Auto fine-tuning
- ⏳ Phase 5: Deployment

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, SQLAlchemy
- **Database**: SQLite
- **ML**: Hugging Face Transformers (upcoming)
- **OCR**: Tesseract (upcoming)