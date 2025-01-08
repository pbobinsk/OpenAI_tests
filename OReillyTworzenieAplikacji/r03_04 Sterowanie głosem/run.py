import gradio as gr
import whisper
from dotenv import load_dotenv

load_dotenv()
import openai
model = whisper.load_model("base")

def transcribe(file):
    print(file)
    transcription = model.transcribe(file)
    return transcription['text']

prompts = {'START': 'Określ intencję wyrażoną we wprowadzonych danych. Odpowiedz jednym słowem: NAPISZ_EMAIL, PYTANIE, INNE.',
           'PYTANIE': 'Odpowiedz jednym słowem: ODPOWIEDŹ, jeżeli możesz odpowiedzieć na pytanie, WIĘCEJ, jeżeli potrzebujesz więcej informacji, lub INNE, jeżeli nie możesz odpowiedzieć. Odpowiedz tylko jednym słowem.',
           'ODPOWIEDŹ': 'Odpowiedz na pytanie.',
           'WIĘCEJ': 'Poproś o więcej informacji.',
           'INNE': 'Odpowiedz, że nie możesz odpowiedzieć na pytanie lub wykonać polecenia.',
           "NAPISZ_EMAIL": 'Odpowiedz "WIĘCEJ", jeżeli brakuje tematu, adresu odbiorcy lub treści. Jeżeli masz wszystkie informacje, odpowiedz "AKCJA_NAPISZ_EMAIL | temat:temat, odbiorca:odbiorca, treść:treść".'}
actions = {'AKCJA_NAPISZ_EMAIL': 'Wiadomość została wysłana. Teraz powiedz, używając języka naturalnego, że akcja została wykonana.'}
messages = [{"role": "user", "content": prompts['START']}]

def generate_answer(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return (response['choices'][0]['message']['content'])

def start(user_input):
    messages.append({"role": "user", "content": user_input})
    return discussion(messages, 'START')

def discussion(messages, last_step):
    answer = generate_answer(messages)
    print(answer)
    if answer in prompts.keys():
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": prompts[answer]})
        return discussion(messages, answer)
    elif answer.split("|")[0].strip() in actions.keys():
        return do_action(answer)
    else:
        if last_step != 'WIĘCEJ':
            messages = []
        last_step = 'KONIEC'
        return answer

def do_action(answer):
    print("Wykonywanie akcji " + answer)
    messages.append({"role": "assistant", "content": answer})
    action = answer.split("|")[0].strip()
    messages.append({"role": "user", "content": actions[action]})
    return discussion(messages, answer)

def start_chat(file):
    input = transcribe(file)
    print(input)
    return start(input)

gr.Interface(
    theme=gr.themes.Soft(),
    fn=start_chat,
    live=True,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text").launch()
