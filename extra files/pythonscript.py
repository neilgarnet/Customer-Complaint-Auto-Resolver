import google.generativeai as genai
import csv
import time
import os

# --- CONFIGURATION ---
API_KEY = "AIzaSyBmG9W167SEjKo02_b5I02LdxVii-eMMK4"
FILENAME = "indian_customer_complaints.csv"
TARGET_ROWS = 1000  # How many rows you want total
BATCH_SIZE = 20     # Rows per request (Best balance for speed/accuracy)
MODEL_NAME = "gemini-3-flash-preview"

# --- SETUP GEMINI ---
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# --- THE PROMPT ---
# We use a "Few-Shot" prompt to teach the AI exactly what we want.
SYSTEM_PROMPT = """
You are a data generator for an Indian customer support system.
Generate 20 unique customer complaints in strict CSV format.
Do NOT output code blocks, markdown, or headers. Just the raw data rows.

Columns: complaint_id, language, region, complaint_text, translated_text_en, category, sentiment, urgency, order_id, amount_inr, channel, reply

Rules:
1. Language Mix: 40% English, 30% Hindi (Devanagari), 30% Hinglish.
2. Context: Use real Indian cities, Indian names, UPI/Banking/E-commerce scenarios.
3. Tone: Emotional, frustrated, detailed (3-5 sentences).
4. Formatting: Ensure CSV quoting is correct for text with commas.

Examples of style:
Hinglish: "Mera paisa kat gaya par recharge nahi hua...", "My money was deducted...", payment_issue, negative, high...
Hindi: "मैंने कल ऑर्डर किया था...", "I ordered yesterday...", delivery_issue, neutral, medium...
English: "I am extremely disappointed with the service...", "I am extremely disappointed...", customer_service_issue, negative, high...
"""

def generate_batch(start_id):
    """Generates a batch of rows starting from a specific ID."""
    try:
        # Prompt asking for specific rows
        prompt = f"{SYSTEM_PROMPT}\n\nStart ID from: CMP{start_id:05d}. Generate exactly {BATCH_SIZE} rows now."
        
        response = model.generate_content(prompt)
        text_data = response.text.strip()
        
        # Simple cleanup to remove any markdown code blocks if the AI adds them
        text_data = text_data.replace("```csv", "").replace("```", "").strip()
        
        # Parse into list of lists to check validity
        rows = []
        reader = csv.reader(text_data.splitlines())
        for row in reader:
            if len(row) >= 10:  # Basic sanity check
                rows.append(row)
        return rows

    except Exception as e:
        print(f"  ⚠️ Error generating batch: {e}")
        return []

def main():
    # 1. Create CSV File with Headers (if it doesn't exist)
    if not os.path.exists(FILENAME):
        with open(FILENAME, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["complaint_id","language","region","complaint_text","translated_text_en","category","sentiment","urgency","order_id","amount_inr","channel","reply"])
    
    # 2. Check how many rows we already have
    with open(FILENAME, "r", encoding="utf-8") as f:
        existing_rows = sum(1 for _ in f) - 1 # Subtract header
    
    print(f"🚀 Starting Generation. Target: {TARGET_ROWS} rows.")
    print(f"📂 Saving to: {FILENAME}")
    print(f"📊 Current Progress: {existing_rows}/{TARGET_ROWS}")

    # 3. Main Loop
    current_count = existing_rows
    while current_count < TARGET_ROWS:
        print(f"  ...Generating batch starting at ID {current_count + 1}...")
        
        new_rows = generate_batch(current_count + 1)
        
        if new_rows:
            # Append to file immediately
            with open(FILENAME, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(new_rows)
            
            current_count += len(new_rows)
            print(f"  ✅ Saved {len(new_rows)} rows. Total: {current_count}/{TARGET_ROWS}")
        else:
            print("  ⚠️ Retrying batch...")
        
        # CRITICAL: Sleep to respect Rate Limits (Free tier limit is usually 15 RPM)
        time.sleep(4) 

    print("\n🎉 DONE! Dataset generation complete.")

if __name__ == "__main__":
    main()