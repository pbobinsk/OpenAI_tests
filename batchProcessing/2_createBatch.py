from openai import OpenAI
client = OpenAI()

batch_input_file_id = 'file-4S5tTgVoJhuZQgKWg2MMQw'
batch = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "nightly eval job"
    }
)
print(batch)
# Batch(id='batch_677e6bb20bcc8190850eea8c04a9e365', completion_window='24h', created_at=1736338354, endpoint='/v1/chat/completions', input_file_id='file-4S5tTgVoJhuZQgKWg2MMQw', object='batch', status='validating', cancelled_at=None, cancelling_at=None, completed_at=None, error_file_id=None, errors=None, expired_at=None, expires_at=1736424754, failed_at=None, finalizing_at=None, in_progress_at=None, metadata={'description': 'nightly eval job'}, output_file_id=None, request_counts=BatchRequestCounts(completed=0, failed=0, total=0))