from dotenv import load_dotenv

load_dotenv()
import openai

# Odwołanie do punktu końcowego openai.Moderation z użyciem modelu text-moderation-latest
response = openai.moderations.create(
    model="text-moderation-latest",
    input="Chcę zabić sąsiada.",
)

# Wyodrębnienie odpowiedzi
print(response.to_json())