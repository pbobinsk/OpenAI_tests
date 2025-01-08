from openai import OpenAI
client = OpenAI()

# client.batches.cancel("batch_abc123")


batches = client.batches.list(limit=10)
print(batches.data)