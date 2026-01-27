# Author: TK
# Date: 27-01-2026
# Desc: Helper for LLM

import os
from dotenv import load_dotenv
from mistral import Mistral

SYSTEM_PROMPT = """You convert user plot requests into safe Python expressions using x.
Return ONLY expressions, one per line. No prose, no code blocks.

Rules:
- Use only: x, numbers, + - * / **, parentheses
- Allowed functions: math.<fn> or np.<fn> (sin, cos, tan, exp, log, sqrt, abs, etc.)
- Use ** for powers (not ^)
- If user asks for multiple curves, output multiple lines (one expression per line)
- Do NOT output assignments, imports, list comprehensions, or any extra text.
Examples:
User: "plot a sine wave" -> math.sin(x)
User: "plot x squared and cosine" -> x**2\nmath.cos(x)
"""

def nl_to_exprs(user_request: str, model: str = "mistral-small-latest") -> str:
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        raise ValueError("Missing MISTRAL_API_KEY in .env")

    user_request = (user_request or "").strip()
    if not user_request:
        raise ValueError("Request is empty")

    with Mistral(api_key=api_key) as client:
        res = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_request},
            ],
            stream=False,
        )


    text = res.choices[0].message.content.strip()
    return text
