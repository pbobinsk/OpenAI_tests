from dotenv import load_dotenv

load_dotenv()
import openai

# Odczytanie transkrypcji z pliku 
with open("transkrypcja.txt", "r") as f:
    transcript = f.read()

# Odwołanie do punktu końcowego openai.ChatCompletion modelu ChatGPT
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Jesteś pomocnym asystentem."},
        {"role": "user", "content": "Streść podany tekst."},
        {"role": "assistant", "content": "Dobrze."},
        {"role": "user", "content": transcript},
    ],
)

print(response["choices"][0]["message"]["content"])