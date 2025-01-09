import gradio as gr
import wave
import os
from scipy.io.wavfile import write
import numpy as np
import tempfile
import shutil

# Funkcja zapisująca plik audio
def save_audio(file_name, audio):
    if audio is None or not file_name.strip():
        return "Nie podano nazwy pliku lub brak audio!"
    
    # Rozpakowanie danych audio (numpy array + sample rate)
    sample_rate, audio_data = audio

    print(sample_rate)
    print(audio)

    # # Normalizacja audio i zapis do pliku WAV
    # output_file = f"{file_name}.wav"
    # write(output_file, sample_rate, (audio_data).astype(np.int16))  # Zapis jako 16-bit WAV
    

    # Generowanie ścieżki 
    #temp_dir = tempfile.gettempdir()
    #output_file = f"{file_name}.wav"
    
    # Kopiowanie pliku audio do ścieżki tymczasowej z nową nazwą
    #shutil.copy(audio, output_file)

    # Normalizacja audio (numpy -> int16)
    audio_data = (audio_data).astype(np.int16)

    # Zapis do pliku WAV
    output_file = f"{file_name}.wav"
    with wave.open(output_file, "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return f"Audio zapisano jako: {output_file}"

# Interfejs Gradio
interface = gr.Interface(
    fn=save_audio,
    inputs=[
        gr.Textbox(label="Podaj nazwę pliku (bez rozszerzenia)"),
        gr.Audio(sources=["microphone"], type="numpy", label="Nagrywaj audio") #numpyfilepath
    ],
    outputs="text",
    title="Nagrywanie i zapisywanie audio",
    description="Nagrywaj audio za pomocą mikrofonu i zapisuj je na dysku pod wskazaną nazwą."
)

# Uruchomienie aplikacji
interface.launch()
