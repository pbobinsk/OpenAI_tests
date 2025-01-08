from dotenv import load_dotenv

load_dotenv()
import openai
from typing import List

def ask_chatgpt(messages):
    response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
    )
    print(response.usage.total_tokens)
    return (response.to_dict()['choices'][0]['message']['content'])

prompt_role = "Jesteś asystentem dziennikarza. \
    Twoim zadaniem jest pisanie artykułów w oparciu o podane FAKTY. \
    Przestrzegaj następujących instrukcji: TON, DŁUGOŚĆ i STYL."

def assist_journalist(facts: List[str], tone: str, length_words: int, style: str):
    facts = ", ".join(facts)
    prompt = f'{prompt_role}\nFAKTY: {facts}\nTON: {tone}\nDŁUGOŚĆ: {length_words} słów\nSTYL: {style}'
    return ask_chatgpt([{"role": "user", "content": prompt}])

print(
    assist_journalist(
        facts=[
            "W zeszłym tygodniu wydano książkę o modelu ChatGPT.",
            "Jej tytuł to Tworzenie aplikacji opartych na modelach GPT-4 i ChatGPT'.",
            "Wydawnictwo Helion.",
        ],
        tone="entuzjazm",
        length_words=50,
        style="wiadomości",
    )
)
