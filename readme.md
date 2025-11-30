Bill Extraction API Solution

This is a Python-based solution using FastAPI and Google Gemini 1.5 Flash to extract line items from complex invoices (PDFs or Images).

Prerequisites

Python 3.9+ installed.

Google Cloud API Key (AI Studio) for Gemini. Get one here.

Installation

Clone this repository.

Install dependencies:

pip install -r requirements.txt


Configuration

Set your Google API Key as an environment variable.

Linux/Mac:

export GOOGLE_API_KEY="your_actual_api_key_here"


Windows (PowerShell):

$env:GOOGLE_API_KEY="your_actual_api_key_here"


Running the Server

Run the server using Uvicorn:

python main.py


The server will start at http://0.0.0.0:8000.

API Usage

Endpoint: POST /extract-bill-data

Request Body:

{
    "document": "[https://hackrx.blob.core.windows.net/assets/datathon-IIT/sample_2.png](https://hackrx.blob.core.windows.net/assets/datathon-IIT/sample_2.png)?..."
}


Response:
Returns the JSON structure specified in the problem statement, including:

Token usage stats.

Page-wise line item breakdown.

Categorization of page types (Pharmacy, Bill Detail, etc.).

Methodology & Logic

Data Ingestion: The API downloads the file from the provided URL into memory.

Multimodal Processing: Instead of using traditional OCR (which loses layout context), we use Gemini 1.5 Flash. This model is multimodal, meaning it can "see" the PDF/Image directly without converting to text first. This is crucial for distinguishing between "Summary" tables and "Detail" tables to avoid double-counting.

Structured Output: We enforce a JSON Schema on the LLM output. This guarantees that the API always returns valid JSON that matches the strict format required by the frontend/evaluator.

Cost Efficiency: Gemini 1.5 Flash is selected over Pro or GPT-4o because it is significantly faster and cheaper, while still maintaining high accuracy for document extraction tasks.