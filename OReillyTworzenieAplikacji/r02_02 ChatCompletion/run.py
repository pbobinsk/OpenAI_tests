from dotenv import load_dotenv

load_dotenv()
import openai

# W modelu GPT 3.5 Turbo punkt końcowy ma nazwę ChatCompletion.
response = openai.chat.completions.create(
    # Model GPT 3.5 Turbo ma nazwę "gpt-3.5-turbo".
    model="gpt-3.5-turbo",
    # Konwersacja jest listą komunikatów.
    messages=[
        {"role": "system", "content": "Jesteś dobrym nauczycielem."},
        {
            "role": "user",
            "content": "Czy istnieją inne wskaźniki złożoności algorytmu niż czas?",
        },
        {
            "role": "assistant",
            "content": "Tak, oprócz złożoności czasowej algorytmu istnieją \
            inne miary, na przykład złożoność przestrzenna.",
        },
        {"role": "user", "content": "Co to jest?"},
    ],
)

#zamiana completion na słownik
print (response.to_dict()["choices"][0]["message"]["content"])
print(response.usage.total_tokens)