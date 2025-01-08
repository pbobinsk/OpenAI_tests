import openai

class ResponseService():
     def __init__(self):
        pass
     
     def generate_response(self, facts, user_question):
         # Odwołanie do punktu końcowego openai.ChatCompletion
         response = openai.ChatCompletion.create(
         model="gpt-3.5-turbo",
         messages=[
               {"role": "user", "content": f"""Uwzględniając FAKTY odpowiedz na PYTANIE.
                PYTANIE: {user_question}. FAKTY: {facts}"""}
            ]
         )

         # Wyodrębnienie odpowiedzi
         return (response['choices'][0]['message']['content'])