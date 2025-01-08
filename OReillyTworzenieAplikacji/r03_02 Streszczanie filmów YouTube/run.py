from dotenv import load_dotenv

load_dotenv()
import openai

# Odczytanie transkrypcji z pliku 
with open("OReillyTworzenieAplikacji/r03_02 Streszczanie filmów YouTube/transkrypcja.txt", "r") as f:
    transcript = f.read()

# Odwołanie do punktu końcowego openai.ChatCompletion modelu ChatGPT
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Jesteś pomocnym asystentem."},
        {"role": "user", "content": "Streść podany tekst."},
        {"role": "assistant", "content": "Dobrze."},
        {"role": "user", "content": transcript},
    ],
)
print(response.usage.total_tokens)
print(response.to_dict()["choices"][0]["message"]["content"])