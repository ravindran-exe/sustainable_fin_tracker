# 🌱 Sustainable Finance Analyzer

Transform your bank statements into actionable sustainability and financial insights with this AI-powered Streamlit application.

---

## Overview

**Sustainable Finance Analyzer** is a user-friendly web app that analyzes your bank statement PDFs to provide:
- Smart transaction categorization
- Carbon footprint tracking
- Financial health dashboards
- AI-powered recommendations for greener, smarter spending

All processing is done locally, and your data remains secure.

---

## Features

- **📊 Transaction Analysis:**  
  Automatic extraction and categorization of transactions, income vs. expense tracking, and monthly summaries.

- **🌍 Carbon Footprint Tracking:**  
  Calculates CO₂ emissions from your spending, assigns a sustainability score, and highlights green vs. high-carbon spending.

- **📈 Comprehensive Dashboards:**  
  Interactive visualizations for cash flow, category spending, carbon trends, and more.

- **🤖 AI-Powered Parsing:**  
  Uses Google Gemini AI for robust transaction extraction, with regex fallback for tricky PDFs.


---

## How It Works

1. **Upload** your bank statement PDF (supports both digital and scanned PDFs).
2. **Choose** processing options (OCR for scanned documents).
3. **Analyze** your financial and environmental impact through interactive dashboards.
4. **Download** your processed data as CSV or JSON.

---

## Technical Architecture

- **Streamlit** for the web interface
- **pdfplumber** and **pytesseract** for PDF and OCR text extraction
- **Google Gemini AI** for advanced transaction parsing
- **Pandas** and **NumPy** for data processing
- **Plotly** for interactive visualizations
- **Modular Python codebase** for easy maintenance and extension

### Main Modules

- `carbon_calculation.py`: Calculates carbon emissions per transaction and aggregates sustainability metrics.
- `categorization.py`: Assigns each transaction to a financial and sustainability category using keyword rules.
- `parsing.py`: Extracts transactions from raw text using AI and regex, with data cleaning and validation.
- `text_extraction.py`: Extracts text and tables from PDFs, with OCR support for scanned documents.
- `utils.py`: Helper functions for data cleaning, JSON conversion, and chunking.
- `visualization.py`: Generates all dashboards and charts using Plotly.

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd sustainable_fin_tracker
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up your Google Gemini API key:**
   - Add your API key to Streamlit secrets:
     ```
     [secrets]
     GOOGLE_API_KEY = "your-google-api-key"
     ```

4. **Run the app:**
   ```sh
   streamlit run main.py
   ```

---

## Requirements

- Python 3.8+
- See `requirements.txt` for all Python dependencies:
  - streamlit
  - pdfplumber
  - pytesseract
  - Pillow
  - numpy
  - pandas
  - plotly
  - google-generativeai
  - scikit-learn

---

## Usage

- Open the app in your browser after running.
- Use the sidebar to upload your PDF and select options.
- Explore the tabs for transaction analysis, carbon dashboard, financial insights, and raw data.
- Download your results as CSV or JSON.

---



## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/)
- [Google Generative AI](https://ai.google.dev/)
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [pytesseract](https://github.com/madmaze/pytesseract)

---

**Empower your finances. Empower the planet. 🌍💸**
