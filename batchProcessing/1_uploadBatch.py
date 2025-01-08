from openai import OpenAI
client = OpenAI()

batch_input_file = client.files.create(
    file=open("batchProcessing/batchinput.jsonl", "rb"),
    purpose="batch"
)

print(batch_input_file)
#FileObject(id='file-4S5tTgVoJhuZQgKWg2MMQw', bytes=517, created_at=1736338133, filename='batchinput.jsonl', object='file', purpose='batch', status='processed', status_details=None)
