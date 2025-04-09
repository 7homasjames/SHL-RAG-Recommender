import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=genai_api_key)

# Load Gemini model
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")


def parse_with_gemini(dom_chunks, slug):
    parsed_results = []

    for i, chunk in enumerate(dom_chunks, start=1):
        prompt = f"""
You are helping HR professionals identify relevant SHL Assessments.

From the SHL job description text below, do the following:

1. Suggest the **most likely job title** if it's not obvious from the slug: `{slug}`.
2. Recommend up to 10 individual SHL Assessments for this role.
3. Output the result as a **JSON array**, where each item has these keys:
   - assessment_name
   - url
   - remote_testing (Yes/No)
   - adaptive_support (Yes/No)
   - duration (e.g., "20 mins")
   - test_type (e.g., "Cognitive", "Personality", etc.)

Only return valid JSON.

SHL Job Description Text:
{chunk}
"""
    
        response = model.generate_content(prompt)
        content_text = response.text.strip()
        print(f"Parsed batch: {i} of {len(dom_chunks)}")
        print("Parsed Response:", content_text)
        parsed_results.append(content_text)
    
    return parsed_results[0] if parsed_results else "[]"
