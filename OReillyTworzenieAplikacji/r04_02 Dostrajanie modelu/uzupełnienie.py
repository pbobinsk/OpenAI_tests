import openai

result = openai.Completion.create(
    model="davinci:ft-personal:marketing-bezpośredni-2023-11-13-12-42-33",
    prompt="Hotel, Nowy Jork, mały ->",
    max_tokens=100,
    temperature=0,
    stop="END"
)

print(result)