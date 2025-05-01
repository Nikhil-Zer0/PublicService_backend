import os
import google.generativeai as genai
import dotenv

dotenv.load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")

def generate_response(new_feedback: str, historical_feedbacks: list):
    history_str = "\n".join(f"- {fb}" for fb in historical_feedbacks) if historical_feedbacks else "No similar past feedbacks"
    
    prompt = f"""
You are a government service feedback analyst. Use the new feedback and only relevant historical feedbacks (same service type or very closely related issues) to craft a very brief response.

New Feedback:
{new_feedback}

Relevant Historical Feedbacks:
{history_str}

Instructions:
- ONLY refer to historical feedbacks that are clearly about the SAME service type or issue.
- DO NOT mix unrelated topics (e.g., if the user mentions water supply, do not refer to garbage disposal).
- DO NOT invent or assume any new details (no fake contact info, plans, phone numbers, etc.).
- If no exact historical match is found, just acknowledge the user's feedback and assure a review.
- Keep the response polite, formal, and professional.

Response:
"""
    res = model.generate_content(prompt)
    return res.text.strip()

def generate_summary(main_feedbacks: list, similar_feedbacks: list):
    main_str = "\n".join(f"- {fb}" for fb in main_feedbacks)
    similar_str = "\n".join(f"- {fb}" for fb in similar_feedbacks) if similar_feedbacks else "None"
    
    prompt = f"""
Analyze these main feedbacks and related similar feedbacks to create a comprehensive summary:

Main Feedbacks:
{main_str}

Related Similar Feedbacks:
{similar_str}

Summarize in 3-5 bullet points:
- Key common issues
- Recurring positive feedback
- Critical areas needing improvement
- Suggested priority actions

Use concise professional language suitable for government reports:
"""
    res = model.generate_content(prompt)
    return res.text.strip()