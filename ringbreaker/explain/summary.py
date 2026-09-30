import os
import json
import google.generativeai as genai

def generate_case_summary(case_file_facts: dict) -> str:
    """
    F17: LLM case summary.
    Generates a short narrative built only from extracted facts present in the case file.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Summary unavailable: GEMINI_API_KEY environment variable not set."

    genai.configure(api_key=api_key)
    
    # Using flash for fast inference during the live stream demo
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are a fraud analyst assistant for RingBreaker.
    Write a short narrative (1-2 sentences) summarizing why this payment was flagged.
    
    CRITICAL CONSTRAINT: You must build this summary ONLY from the facts extracted below.
    Cite only the fields present in the case file. Do not hallucinate, invent names, or add outside context.
    
    Case File Facts:
    {json.dumps(case_file_facts, indent=2)}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Summary generation failed. Fallback: Payment flagged based on pattern rules. ({str(e)})"