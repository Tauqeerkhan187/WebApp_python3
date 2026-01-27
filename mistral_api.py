# Author:TK
# Date: 27-01-2026
# Loads mistral ai into our webapp

import os
from dotenv import load_dotenv
from mistralai import Mistral

def main():
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY", "")

    if not api_key:
        raise SystemExit("Missing MISTRAL_API_KEY. Put it in .env")

    with Mistral(api_key=api_key) as client:
        res = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "user", "content": "Say hello in one short sentence."}
            ],

            stream=False,
        )

        print(res.choices[0].message.content)

if __name__ == "__main__":
    main()

