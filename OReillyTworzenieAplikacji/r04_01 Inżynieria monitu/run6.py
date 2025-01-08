import openai

def chat_completion(prompt, model="gpt-4", temperature=0):
    res = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    print(res["choices"][0]["message"]["content"])

prompt = """
Rola: jesteś ekspertem opracowującym zdrowe diety dla sportowców wyczynowych. Zajmujesz się odżywianiem niezbędnym do szybkiego powrotu do zdrowia.
Kontekst: uprawiam sport przez 2 godziny dziennie. Jestem wegetarianinem i nie lubię zielonych warzyw. Świadomie odżywiam się zdrowo.
Zadanie: opierając się na swojej wiedzy, zaproponuj mi danie główne na dzisiejszy obiad. Potrzebuję tabeli z dwiema kolumnami. Wiersze niech zawierają składniki dania głównego. Pierwsza kolumna tabeli to nazwa składnika, a druga to liczba gramów składnika dla jednej osoby. Nie podawaj przepisu na przygotowanie dania głównego.
"""
chat_completion(prompt)