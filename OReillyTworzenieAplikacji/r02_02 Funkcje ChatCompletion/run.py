from dotenv import load_dotenv

load_dotenv()
import openai
import json

def find_product(sql_query):
    # Wysłanie zapytania do bazy
    results = [
        {"name": "pióro", "color": "niebieskie", "price": 199},
        {"name": "pióro", "color": "czerwone", "price": 178},
    ]
    return results

functions = [
    {
        "name": "find_product",
        "description": "Utworzenie listy produktów na podstawie zapytania SQL",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "Zapytanie SQL",
                }
            },
            "required": ["sql_query"],
        },
    }
]

def run(user_question):
    # Wysłanie pytania i funkcji do modelu GPT
    messages = [{"role": "user", "content": user_question}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613", messages=messages, functions=functions
    )
    response_message = response["choices"][0]["message"]

	  # Dołączenie odpowiedzi asystenta do komunikatu
    print(response_message)
    messages.append(response_message)
    

	  # Wywołanie funkcji i dodanie wyniku do komunikatów
    if response_message.get("function_call"):
        function_name = response_message["function_call"]["name"]
        if function_name == "find_product":
            function_args = json.loads(
                response_message["function_call"]["arguments"]
            )
            products = find_product(function_args.get("sql_query"))
        else:
            # Obsługa błędów
            products = []
		    # Dołączenie wyniku funkcji do komunikatów
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": json.dumps(products),
            }
        )
		    # Uzyskanie nowej odpowiedzi z modelu GPT, aby pretłumaczyć wynik funkcji na język naturalny
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )
        return second_response


print(run("Potrzebuję dwóch najlepszych produktów w cenie do 200 zł"))
