from dotenv import load_dotenv

load_dotenv()
import openai

# Odwołanie do punktu końcowego openai.Completion
response = openai.Completion.create(
    model="text-davinci-003", prompt="Witaj, świecie!"
)

# Wyodrębnienie odpowiedzi
print(response["choices"][0]["text"])
