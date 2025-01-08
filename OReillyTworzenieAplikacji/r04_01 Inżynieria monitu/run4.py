import openai

def chat_completion(prompt, model="gpt-4", temperature=0):
    res = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    print(res["choices"][0]["message"]["content"])

prompt = """
Kontekst: uprawiam sport przez 2 godziny dziennie. Jestem wegetarianinem i nie lubię zielonych warzyw. Świadomie odżywiam się zdrowo.
Zadanie: zaproponuj  mi danie główne na dzisiejszy obiad.
Nie wykonuj tego zadania! Zamiast tego zadaj mi pytania dotyczące kontekstu, abyś po udzieleniu przeze mnie odpowiedzi mógł lepiej wykonać to zadanie.
"""
chat_completion(prompt)
