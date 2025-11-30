
import os
import json
import requests
import uvicorn
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import google.generativeai as genai

# --- Configuration ---
# Load API Key from environment variable
GOOGLE_API_KEY = ""

if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY is not set. The API will fail until this is set.")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Model
model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI(title="Bill Extraction API")

# --- Pydantic Models ---

class BillRequest(BaseModel):
    document: str = Field(..., description="URL of the document (Image or PDF)")

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float

class PageLineItems(BaseModel):
    page_no: str
    page_type: Literal["Bill Detail", "Final Bill", "Pharmacy", "Unknown"]
    bill_items: List[BillItem]

class ExtractionData(BaseModel):
    pagewise_line_items: List[PageLineItems]
    total_item_count: int

class APIResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: Optional[ExtractionData] = None
    error: Optional[str] = None

# --- Helper Functions ---

def download_file(url: str) -> bytes:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")

def process_document_with_llm(file_bytes: bytes, mime_type: str) -> dict:
    """
    Sends the file to Gemini and extracts structured data.
    """
    
    # Define the Schema for the LLM to follow strictly
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "pagewise_line_items": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "page_no": {"type": "STRING"},
                        "page_type": {"type": "STRING", "enum": ["Bill Detail", "Final Bill", "Pharmacy", "Unknown"]},
                        "bill_items": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "item_name": {"type": "STRING"},
                                    "item_amount": {"type": "NUMBER"},
                                    "item_rate": {"type": "NUMBER"},
                                    "item_quantity": {"type": "NUMBER"}
                                },
                                "required": ["item_name", "item_amount", "item_rate", "item_quantity"]
                            }
                        }
                    },
                    "required": ["page_no", "page_type", "bill_items"]
                }
            }
        },
        "required": ["pagewise_line_items"]
    }

    prompt = """
    You are an expert autonomous data extraction agent for medical and retail invoices. 
    Analyze the provided document (which may contain multiple pages) and extract line item details.

    RULES:
    1. Extract 'item_name', 'item_amount' (net amount), 'item_rate', and 'item_quantity' for every line item.
    2. Identify the 'page_type' based on content:
       - 'Pharmacy' for medicine lists.
       - 'Bill Detail' for service/room charges.
       - 'Final Bill' for summary pages.
    3. CRITICAL: Do NOT double count. If a page is a 'Summary' of previous pages, extract it but mark it clearly. 
       However, usually, 'Final Bill' pages contain the grand total, while detail pages contain items. 
       Extract items from where they are listed in detail.
    4. If a quantity is not explicitly stated but implied as 1, use 1.
    5. 'item_amount' should be the total for that line (rate * qty - discount).
    """

    try:
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.1
        )

        content_part = {
            "mime_type": mime_type,
            "data": file_bytes
        }

        response = model.generate_content(
            [prompt, content_part],
            generation_config=generation_config
        )
        
        json_output = json.loads(response.text)
        usage = response.usage_metadata
        
        return {
            "result": json_output,
            "usage": {
                "total_tokens": usage.total_token_count,
                "input_tokens": usage.prompt_token_count,
                "output_tokens": usage.candidates_token_count
            }
        }

    except Exception as e:
        print(f"LLM Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document with AI model: {str(e)}")


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Serves a simple HTML form for testing the API easily in a browser.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bill Extractor Tester</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
            input[type="text"] { width: 70%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; }
            button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            #result { margin-top: 20px; white-space: pre-wrap; background: #f4f4f4; padding: 15px; border-radius: 5px; }
            .loading { color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <h1>Bill Extraction API Tester</h1>
        <p>Enter the URL of an invoice/bill (Image or PDF) below:</p>
        <div>
            <input type="text" id="billUrl" placeholder="https://example.com/bill.png">
            <button onclick="extractData()">Extract Data</button>
        </div>
        <div id="result">Results will appear here...</div>

        <script>
            async function extractData() {
                const url = document.getElementById('billUrl').value;
                const resultDiv = document.getElementById('result');
                
                if (!url) {
                    alert("Please enter a URL");
                    return;
                }

                resultDiv.innerHTML = '<span class="loading">Processing... This may take a few seconds.</span>';

                try {
                    const response = await fetch('/extract-bill-data', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ document: url })
                    });

                    const data = await response.json();
                    resultDiv.textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    resultDiv.textContent = 'Error: ' + error.message;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/extract-bill-data", response_model=APIResponse)
async def extract_bill_data(request: BillRequest):
    
    # 1. Download File
    file_bytes = download_file(request.document)
    
    # 2. Determine Mime Type
    if request.document.lower().endswith('.pdf'):
        mime_type = "application/pdf"
    else:
        mime_type = "image/jpeg" # Default to image for safety

    # 3. Process
    try:
        llm_response = process_document_with_llm(file_bytes, mime_type)
        
        raw_data = llm_response["result"]
        usage_data = llm_response["usage"]
        
        # 4. Calculate total item count
        pagewise_items = raw_data.get("pagewise_line_items", [])
        total_count = sum(len(page.get("bill_items", [])) for page in pagewise_items)
        
        return APIResponse(
            is_success=True,
            token_usage=TokenUsage(**usage_data),
            data=ExtractionData(
                pagewise_line_items=pagewise_items,
                total_item_count=total_count
            )
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        return APIResponse(
            is_success=False,
            token_usage=TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0),
            error=str(e)
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)