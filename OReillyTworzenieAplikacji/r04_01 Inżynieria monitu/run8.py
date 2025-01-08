import openai

def chat_completion(prompt, model="gpt-4", temperature=0):
    res = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    print(res["choices"][0]["message"]["content"])

prompt = """
Idę do domu --> 🙂 do 🏠
Mój pies jest smutny --> Mój 🐶 jest ☹️
Biegnę szybko --> 🙂 biegnę ⚡️
Kocham moją żonę --> 🙂 ❤️ moją żonę
Dziwczynka bawi się piłką --> 👩 🎮 🏀
Chłopak pisze list do dziewczyny -->
"""

chat_completion(prompt)