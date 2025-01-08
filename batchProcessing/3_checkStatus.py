from openai import OpenAI
client = OpenAI()

batch = client.batches.retrieve("batch_677e6bb20bcc8190850eea8c04a9e365")
print(batch.status)
print(batch.output_file_id)
