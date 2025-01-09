import gradio as gr
import wave
import numpy as np
import shutil

counter = 1

# Funkcja zapisująca plik audio
def save_audio(file_name, audio_chunk):
    if audio_chunk is None:
        return "Nie wykryto dźwięku"
    
    #sample_rate, audio_data = audio_chunk
    global counter
    print(audio_chunk,counter)
    counter +=1 
    output_file = f"{file_name}_{counter}.wav"
    shutil.copy(audio_chunk, output_file)
    
    #duration = len(audio_data) / sample_rate

    #return (f"Odebrano fragment dźwięku o długości {duration:.2f} sekund")
    return (f"Odebrano fragment dźwięku o długości {audio_chunk} ")


# Interfejs Gradio
interface = gr.Interface(
    fn=save_audio,
    inputs=[
        gr.Textbox(label="Podaj nazwę pliku (bez rozszerzenia)"),
        gr.Audio(sources=["microphone"], type="filepath", label="Nagrywaj audio strumieniowo", streaming=True)#, stream_every=1.0) #numpyfilepath
    ],
    outputs="text",
    title="Nagrywanie i zapisywanie audio",
    live=True,
    description="Nagrywaj audio za pomocą mikrofonu i zapisuj je na dysku pod wskazaną nazwą."
)

# Uruchomienie aplikacji
interface.launch()
