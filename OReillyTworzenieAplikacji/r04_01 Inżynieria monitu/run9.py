import openai

def chat_completion(prompt, model="gpt-4", temperature=0):
    res = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    print(res["choices"][0]["message"]["content"])

prompt = """
Podaj nazwy pięciu zwierząt w formacie JSON. Wynik musi mieć format odpowiedni dla metody json.loads().
"""
chat_completion(prompt, model='gpt-4')