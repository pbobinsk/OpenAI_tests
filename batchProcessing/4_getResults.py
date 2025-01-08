from openai import OpenAI
client = OpenAI()

file_response = client.files.content('file-13o6AMuYmu9VgHcVdvjYkj')
print(file_response.text)