import gradio as gr
import wave
import numpy as np
import shutil, time

counter = 1
buffer = []
last_processed_time = time.time()

# Funkcja zapisująca plik audio
def save_audio(file_name, audio_chunk):
    if audio_chunk is None:
        return "Nie wykryto dźwięku"
    
    #sample_rate, audio_data = audio_chunk
    global counter
    global buffer, last_processed_time

    sample_rate, audio_data = audio_chunk
    buffer.extend(audio_data)

    interval = 1.0  # Przetwarzaj dane co 1 sekundę
    current_time = time.time()
    if current_time - last_processed_time >= interval:
        duration = len(buffer) / sample_rate
        last_processed_time = current_time
        print(audio_chunk,counter)
        print(type(audio_data),len(audio_data))
        print(type(buffer),len(buffer))

        counter +=1 
        output_file = f"{file_name}_{counter}.wav"
        with wave.open(output_file, "w") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            npbuf = np.array(buffer)
            wav_file.writeframes(npbuf.tobytes())
        buffer.clear()

        return f"Przetworzono fragment o długości {duration:.2f} sekund i zapisano w {output_file}"

    return "Oczekiwanie na więcej danych..."


    # print(audio_chunk,counter)
    # counter +=1 
    # output_file = f"{file_name}_{counter}.wav"

    # # shutil.copy(audio_chunk, output_file)
    # sample_rate, audio_data = audio_chunk
    # duration = len(audio_data) / sample_rate

    # with wave.open(output_file, "w") as wav_file:
    #     wav_file.setnchannels(1)  # Mono
    #     wav_file.setsampwidth(2)  # 16-bit
    #     wav_file.setframerate(sample_rate)
    #     wav_file.writeframes(audio_data.tobytes())

    # return (f"Odebrano fragment dźwięku o długości {duration:.2f} sekund")
    #return (f"Odebrano fragment dźwięku o długości {audio_chunk} ")


# Interfejs Gradio
interface = gr.Interface(
    fn=save_audio,
    inputs=[
        gr.Textbox(label="Podaj nazwę pliku (bez rozszerzenia)"),
        gr.Audio(sources=["microphone"], type="numpy", label="Nagrywaj audio strumieniowo", streaming=True)#, stream_every=1.0) #numpyfilepath
    ],
    outputs="text",
    title="Nagrywanie i zapisywanie audio",
    live=True,
    description="Nagrywaj audio za pomocą mikrofonu i zapisuj je na dysku pod wskazaną nazwą."
)

# Uruchomienie aplikacji
interface.launch()
