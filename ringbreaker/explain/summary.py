"""F17: LLM case summary.

Generates a short narrative built only from extracted facts present in the case file.
Uses Gemini when GEMINI_API_KEY is present; provides a concise deterministic
fallback narrative when offline.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict


def generate_case_summary(case_file_facts: Dict[str, Any]) -> str:
    """Generate a 1-2 sentence narrative summarizing why this payment was flagged."""
    risk = float(case_file_facts.get("overall_risk", case_file_facts.get("risk_score", 0.0)))
    risk_pct = float(case_file_facts.get("risk_percent", round(risk * 100.0, 2)))
    action = str(case_file_facts.get("action", "REVIEW")).upper()
    pattern = case_file_facts.get("pattern") or "anomalous velocity/relationship"
    cf = case_file_facts.get("counterfactual") or {}
    dominant = cf.get("dominant_factor", "risk factors")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return (
            f"Flagged for {action} with combined risk {risk:.4f} ({risk_pct:.2f}%). "
            f"Detected topology aligns with {pattern}, with {dominant} acting as the primary driver."
        )

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
You are a fraud analyst assistant for RingBreaker.
Write a short narrative (1-2 sentences) summarizing why this payment was flagged.

CRITICAL CONSTRAINT: You must build this summary ONLY from the facts extracted below.
Cite only the fields present in the case file. Do not hallucinate, invent names, or add outside context.
The overall risk is {risk:.4f} ({risk_pct:.2f}%) with action {action}.

Case File Facts:
{json.dumps(case_file_facts, indent=2, default=str)}
"""
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return (
            f"Flagged for {action} with combined risk {risk:.4f} ({risk_pct:.2f}%). "
            f"Detected topology aligns with {pattern}, with {dominant} acting as the primary driver."
        )