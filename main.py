import os
import json
import requests
import uvicorn
import io
import base64
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import google.generativeai as genai
from PIL import Image

# --- Configuration ---
# ideally load this from .env
# You need a Google Cloud API Key (AI Studio)
GOOGLE_API_KEY = "AIzaSyBbj-lIyOixDypOwRb913RlwU1fK45wUCs"

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Model
# We use Gemini 1.5 Flash for speed and cost-efficiency with high vision accuracy
model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI(title="Bill Extraction API")

# --- Pydantic Models for Request/Response ---

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
    
    # 1. Define the Schema for the LLM to follow strictly
    # This ensures the output is always valid JSON matching our API needs
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "pagewise_line_items": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "page_no": {"type": "STRING"},
                        "page_type": {"type": "STRING", "enum": ["Bill Detail", "Final Bill", "Pharmacy"]},
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

    # 2. Construct the Prompt
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

    # 3. Call Gemini
    # Gemini accepts raw bytes for PDFs and Images if mime_type is specified
    try:
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.1 # Low temperature for factual extraction
        )

        content_part = {
            "mime_type": mime_type,
            "data": file_bytes
        }

        response = model.generate_content(
            [prompt, content_part],
            generation_config=generation_config
        )
        
        # 4. Parse Response
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
        # Fallback logging or handling
        print(f"LLM Error: {e}")
        raise HTTPException(status_code=500, detail="Error processing document with AI model.")


# --- API Endpoint ---

@app.post("/extract-bill-data", response_model=APIResponse)
async def extract_bill_data(request: BillRequest):
    
    # 1. Download File
    file_bytes = download_file(request.document)
    
    # 2. Determine Mime Type (Simple check)
    if request.document.lower().endswith('.pdf'):
        mime_type = "application/pdf"
    elif request.document.lower().endswith(('.png', '.jpg', '.jpeg')):
        mime_type = "image/jpeg" # Gemini handles png/jpg broadly
    else:
        # Fallback detection using magic numbers could go here
        # For this hackathon scope, we assume extension is valid or default to image
        mime_type = "image/jpeg"

    # 3. Process
    try:
        llm_response = process_document_with_llm(file_bytes, mime_type)
        
        raw_data = llm_response["result"]
        usage_data = llm_response["usage"]
        
        # 4. Calculate total item count (Post-processing)
        pagewise_items = raw_data.get("pagewise_line_items", [])
        total_count = sum(len(page.get("bill_items", [])) for page in pagewise_items)
        
        # 5. Construct Final Response
        return APIResponse(
            is_success=True,
            token_usage=TokenUsage(**usage_data),
            data=ExtractionData(
                pagewise_line_items=pagewise_items,
                total_item_count=total_count
            )
        )

    except HTTPException as he:
        # Pass through HTTP exceptions
        raise he
    except Exception as e:
        return APIResponse(
            is_success=False,
            token_usage=TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0),
            error=str(e)
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)