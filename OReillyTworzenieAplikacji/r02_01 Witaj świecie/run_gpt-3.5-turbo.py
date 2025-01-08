from dotenv import load_dotenv

load_dotenv()
import os
import openai

# Sprawdź, czy jest zdefiniowana zmienna środowiskowa OPENAI_API_KEY.

# Wywołaj punkt końcowy ChatCompletion modelu ChatGPT:
response = openai.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "user", "content": "Witaj, świecie!"}
    ]
)

# Wyodrębnij odpowiedź:
print(response.choices[0].message.content)
print(response.usage.to_json())