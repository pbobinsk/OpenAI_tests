import openai

class IntentService():
     def __init__(self):
        pass
     
     def get_intent(self, user_question: str):
         # Odwołanie do punktu końcowego openai.ChatCompletion
         response = openai.ChatCompletion.create(
         model="gpt-3.5-turbo",
         messages=[
               {"role": "user", 
                "content": f"""Wyodrębnij słowa kluczowe z następującego pytania.
                 Nie odpowiadaj, podaj tylko słowa kluczowe. {user_question}"""}
            ]
         )

         # Wyodrębnienie odpowiedzi
         return (response['choices'][0]['message']['content'])