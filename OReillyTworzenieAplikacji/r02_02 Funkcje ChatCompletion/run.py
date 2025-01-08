from dotenv import load_dotenv

load_dotenv()
import openai
import json

def find_product(sql_query):
    # Wysłanie zapytania do bazy
    print('Symulacja wysłania zapytania do bazy')
    print('Zapytanie: '+sql_query)
    results = [
        {"name": "pióro", "color": "niebieskie", "price": 199},
        {"name": "pióro", "color": "czerwone", "price": 178},
    ]
    print('wynik: ')
    print(results)
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
    print('Wysłanie pytania i funkcji do modelu GPT')
    print('Zapytanie: '+user_question)
    print('Funkcja: '+functions[0]["name"])
    messages = [{"role": "user", "content": user_question}]

    response = openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages, functions=functions
    )
    response_message = response.to_dict()["choices"][0]["message"]

    print('Odpowiedź 1, która zostanie dołączona do kolejnego zapytania:')

	  # Dołączenie odpowiedzi asystenta do komunikatu
    print(json.dumps(response_message,indent=2))
    messages.append(response_message)
    
    print('Messeges po dodaniu:')
    print(json.dumps(messages,indent=2))

    print('Wołanie fukcji')

	  # Wywołanie funkcji i dodanie wyniku do komunikatów
    if response_message.get("function_call"):
        function_name = response_message["function_call"]["name"]
        if function_name == "find_product":
            print('Argument zwrócony przez GPT:')
            function_args = json.loads(
                response_message["function_call"]["arguments"]
            )
            print(function_args)
            print('I wołam funkcje z parametrem z sql_query')
            products = find_product(function_args.get("sql_query"))
            print('Wynik:')
            print(products)
        else:
            # Obsługa błędów
            products = []
		    # Dołączenie wyniku funkcji do komunikatów
        print('Wynik wraca do GPT, messages po dodaniu:')
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": json.dumps(products),
            }
        )
        print(json.dumps(messages, indent=2))
		    # Uzyskanie nowej odpowiedzi z modelu GPT, aby pretłumaczyć wynik funkcji na język naturalny
        print('# Uzyskanie nowej odpowiedzi z modelu GPT, aby pretłumaczyć wynik funkcji na język naturalny')    
        second_response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return second_response

completionOut = run("Potrzebuję dwóch najlepszych produktów w cenie do 200 zł")
print(completionOut.choices[0].to_json())
print(completionOut.usage.total_tokens)
