import os
import asyncio
import time
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from pdf2image import convert_from_path
import google.generativeai as genai
import json

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Global configurations
USER_SESSIONS = {}
QUESTION_PROMPT_TEMPLATE = """
        You are an expert in understanding invoices.
        You will receive input images as invoices &
        You are given with the data and user query.
        You just need to answer based on the information provided.
        Answer in a conversational manner, as if talking to a human.
        Any questions outside the information in the invoice will be ignored and not answered.
        Thank you!
            Answer this question based strictly on the invoice content:
                Question: {question}

                Follow these guidelines:
                1. Be specific and use exact values from the invoice when possible
                2. If information is missing/unclear, state "Not clearly specified in the invoice"
                3. Format currency values with their symbols (e.g., ₹, $, €)
                4. Never guess or assume values not shown in the invoice
"""
EXTRACTION_PROMPT = """
               You are an expert in understanding and extracting detailed information from invoices of any kind.
               You will receive various invoice images as input and need to provide accurate and comprehensive answers based on the extracted details.
               The extracted information should cover a wide range of invoices, including food invoices, e-commerce invoices, utility bills, and any other type of invoice.
               Below is an example structure, but please adapt and modify the structure as needed for different invoice types:
               
               {
                   "Supplier": {
                       "Name": "Supplier Name",
                       "Address": "123 Supplier St, City, ZIP",
                       "PAN Number": "PAN123456789",
                       "GST Number": "GST0000",
                       "Contact": "xxxxxxxxxxx"
                   },
                   "ReceiptDetails": {
                       "Billing Address": "Customer Billing Address",
                       "Shipping Address": "Customer Shipping Address",
                       "Invoice Number": "INV123456",
                       "Date": "YYYY-MM-DD",
                       "Time": "HH:MM:SS",
                   },
                   "Items": [
                       {
                           "ItemName": "Product Name or Service",
                           "Quantity": 2,
                           "Unit Price": "50.00",
                           "Total Price": "100.00"
                       },
                       {
                           "ItemName": "Another Product",
                           "Quantity": 3,
                           "Unit Price": "30.00",
                           "Total Price": "90.00"
                       }
                   ],

                   "Payment Details" :{
                       "Payment Method": "Credit Card / Cash / Bank Transfer / Other",
                       "Currency": "USD",
                       "Total Amount": "250.00",
                       "Taxes": "25.00",
                       "Discounts": "5.00"
                   }
                   "Tax calculation" :{
                    ......
                   }
               }

               Instructions for Processing Invoices:

               1. Tax Calculations:
                   Sum all taxes (e.g., CGST, SGST, and any others) and ensure the total tax amount is accurate.
                   If tax percentages are listed, verify they sum up correctly to the total tax amount.
                   Ensure that the taxable amount plus total taxes equals the final amount after applying any discounts.

               2. Item Extraction:
                   Split item details correctly even if item names or descriptions span multiple rows.
                   Treat every line within the same column as part of a single item.
                   If quantity is more than one then unit price and total price can not be same
                   If unit price is not given of an item then divide amount by quantity to get Unit price

               3. Invoice Generation Time:
                   Look for common formats such as "am/PM" to identify when the invoice was generated or paid.

               4. Data Verification:
                   Cross-verify extracted data to ensure it matches the expected invoice totals:
                     Verify if the total amount equals the sum of item prices plus taxes, minus any discounts.
                     Check if the total quantity of items matches the sum of quantities for each item if provided.

               5. Subtotal Calculation:
                   Ignore extracting 'Subtotal' directly from the image.
                   The actual 'Subtotal' should be calculated as:
                    Subtotal = TotalAmount - TaxAmount + DiscountAmount`
                   Ensure the validation that:
                    FinalTotal = Subtotal + TaxAmount - DiscountAmount`

               6. Currency Extraction:
                  Don't just use the currency sumbol in json as it is. Find out the name of that currency adn then add that name in currency.
               """


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command."""
    await update.message.reply_text(
        "Hi! I'm your Invoice Assistant. Please send me an invoice file (image or PDF) to get started."
    )


async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles file uploads and initiates processing."""
    user = update.message.from_user
    document = update.message.document
    
    if not document.file_name.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
        await update.message.reply_text("⚠️ Unsupported file format! Please upload a PDF or image.")
        return

    # Create user directory
    user_dir = f"user_{user.id}"
    os.makedirs(user_dir, exist_ok=True)

    # Download file
    file = await document.get_file()
    file_path = os.path.join(user_dir, document.file_name)
    await file.download_to_drive(file_path)
    
    await update.message.reply_text("🔄 Processing your document...")

    # Process PDF files
    if document.file_name.lower().endswith('.pdf'):
        file_path = await convert_pdf_to_image(file_path, user_dir)
        if not file_path:
            await update.message.reply_text("❌ Failed to process PDF.")
            return

    # Store file path and process
    USER_SESSIONS[user.id] = {"image_path": file_path}
    await process_invoice_image(update, file_path, user.id)


async def convert_pdf_to_image(pdf_path: str, output_dir: str) -> str:
    """Converts PDF to JPEG image using threads."""
    try:
        images = await asyncio.to_thread(convert_from_path, pdf_path)
        image_path = f"{output_dir}/invoice.jpg"
        await asyncio.to_thread(images[0].save, image_path, "JPEG")
        return image_path
    except Exception as e:
        print(f"PDF conversion error: {e}")
        return None


async def process_invoice_image(update: Update, image_path: str, user_id: int) -> None:
    """Processes the invoice image, stores the result, and sends a JSON file."""
    try:
        response = await asyncio.to_thread(
            genai.GenerativeModel('gemini-1.5-flash').generate_content,
            [EXTRACTION_PROMPT, {"mime_type": "image/jpeg", "data": open(image_path, "rb").read()}]
        )
        extracted_text = response.text
        
        # Clean and validate JSON response
        cleaned_json = extracted_text.strip('```').replace('json\n', '', 1).strip()
        
        try:
            json_data = json.loads(cleaned_json)
            user_dir = f"user_{user_id}"
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            json_filename = f"{base_name}_data.json"
            json_path = os.path.join(user_dir, json_filename)
            
            with open(json_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4, ensure_ascii=False)
            
            
            await update.message.reply_text("✅ Extraction complete! Here's the JSON data:")
            await update.message.reply_text(f"{extracted_text}",)
            await update.message.reply_text("✅ Here's the JSON File:")
            await update.message.reply_document(document=open(json_path, 'rb'), filename=json_filename)
            await update.message.reply_text("💡 You can now ask questions using /ask")
            
        except json.JSONDecodeError:
            await update.message.reply_text("⚠️ The response wasn't valid JSON. Here's the raw text:")
            await update.message.reply_text(extracted_text)
            
    except Exception as e:
        await update.message.reply_text(f"❌ Processing error: {str(e)}")


async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles invoice-related questions using structured prompt."""
    user = update.message.from_user
    question = " ".join(context.args)

    if not USER_SESSIONS.get(user.id, {}).get("image_path"):
        await update.message.reply_text("⚠️ Please upload an invoice first!")
        return

    try:
        full_prompt = QUESTION_PROMPT_TEMPLATE.format(question=question)
        response = await asyncio.to_thread(
            genai.GenerativeModel('gemini-1.5-flash').generate_content,
            [full_prompt, 
             {"mime_type": "image/jpeg", 
              "data": open(USER_SESSIONS[user.id]["image_path"], "rb").read()}]
        )
        await update.message.reply_text(f"📝 Response:\n\n{response.text}")
        await update.message.reply_text("💡 Ask another question or /end to finish")
    except Exception as e:
        await update.message.reply_text(f"❌ Error answering question: {str(e)}")

async def end_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Cleans up conversation resources."""
    user = update.message.from_user
    USER_SESSIONS.pop(user.id, None)
    await update.message.reply_text("💬 Session ended. Start again with /start.")


def main() -> None:
    """Main application setup."""
    application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
    
    # Register handlers
    handlers = [
        CommandHandler('start', start),
        CommandHandler('ask', ask_question),
        CommandHandler('end', end_conversation),
        MessageHandler(filters.Document.ALL, handle_file)
    ]
    
    for handler in handlers:
        application.add_handler(handler)

    application.run_polling()


if __name__ == "__main__":
    main()