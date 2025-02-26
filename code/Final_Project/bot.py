import os
import tempfile
import streamlit as st
from pdf2image import convert_from_path
import google.generativeai as genai
import json
import re

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define prompts (same as original)
QUESTION_PROMPT_TEMPLATE = """..."""  # Keep your existing template
EXTRACTION_PROMPT = """..."""  # Keep your existing template

def convert_pdf_to_image(pdf_path: str):
    """Convert PDF to JPEG image"""
    try:
        images = convert_from_path(pdf_path)
        return images[0] if images else None
    except Exception as e:
        st.error(f"PDF conversion error: {e}")
        return None

def clean_json_response(response: str):
    """Clean the response to extract valid JSON."""
    # Remove markdown code blocks (e.g., ```json)
    response = re.sub(r'```json|```', '', response).strip()

    # Remove extra text before or after JSON
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return None

def process_invoice(image_path: str):
    """Process invoice using Gemini and extract JSON data."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            [EXTRACTION_PROMPT, {"mime_type": "image/jpeg", "data": open(image_path, "rb").read()}]
        )
        extracted_text = response.text

        # Clean the response to extract JSON
        cleaned_json = clean_json_response(extracted_text)

        # Validate and parse JSON
        if cleaned_json:
            try:
                json_data = json.loads(cleaned_json)
                return json_data
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse JSON: {e}")
                st.write("Raw response:", extracted_text)
                return None
        else:
            st.error("No valid JSON found in the response.")
            st.write("Raw response:", extracted_text)
            return None

    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None

def ask_question(image_path: str, question: str):
    """Handle questions using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        full_prompt = QUESTION_PROMPT_TEMPLATE.format(question=question)
        response = model.generate_content(
            [full_prompt, {"mime_type": "image/jpeg", "data": open(image_path, "rb").read()}]
        )
        return response.text
    except Exception as e:
        st.error(f"Question error: {str(e)}")
        return None

def main():
    st.title("Invoice Processing Assistant")
    st.write("Upload an invoice (PDF or image) to get started")

    # File upload section
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            file_path = tmp_file.name
            tmp_file.write(uploaded_file.getvalue())

        # Handle PDF files
        if uploaded_file.type == "application/pdf":
            image = convert_pdf_to_image(file_path)
            if image:
                # Save converted image
                image_path = f"{file_path}.jpg"
                image.save(image_path)
                st.image(image, caption="Converted Invoice Image")
            else:
                st.error("Failed to process PDF")
                return
        else:
            image_path = file_path
            st.image(uploaded_file.getvalue(), caption="Uploaded Invoice")

        # Store in session state
        st.session_state.image_path = image_path

        # Process invoice
        if 'processed_data' not in st.session_state:
            with st.spinner("Processing invoice..."):
                processed_data = process_invoice(image_path)
                if processed_data:
                    st.session_state.processed_data = processed_data

        # Show processed data
        if 'processed_data' in st.session_state:
            st.subheader("Extracted Data")
            st.json(st.session_state.processed_data)

            # Question handling
            st.subheader("Ask a Question")
            question = st.text_input("Enter your question about the invoice:")
            
            if question:
                with st.spinner("Analyzing..."):
                    answer = ask_question(st.session_state.image_path, question)
                    if answer:
                        st.write("**Answer:**")
                        st.write(answer)

if __name__ == "__main__":
    main()