# 🤖 Smart Invoice AI System

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**AI-powered invoice processing system that learns from your corrections and improves over time.**

## ✨ Features

- 📄 **Multi-format Support**: PDF, PNG, JPG, CSV
- 🧠 **Real-time Learning**: AI improves from user corrections
- 🎯 **Smart Extraction**: 9 key fields with confidence scores
- 📊 **Analytics Dashboard**: Track accuracy and improvements
- 🔒 **Self-contained**: No external APIs required

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

### Installation
```bash
git clone https://github.com/yourusername/smart-invoice-ai.git
cd smart-invoice-ai
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
