import sounddevice as sd
from pynput import keyboard
import numpy as np
import wave
import time
import threading
import openai
import tempfile
import os
import pyautogui
import configparser

# Read configuration from file
config = configparser.ConfigParser()
config.read("config.ini")

api_key = config.get("OpenAI", "api_key")
auto_enter = config.getboolean("Settings", "auto_enter")

openai.api_key = api_key

# Global variables to manage recording state
is_recording = False
audio_data = np.array([])
start_time = 0
fs = 44100  # Sample rate
min_recording_duration = 0.5  # Minimum recording duration (in seconds)


def on_press(key):
    global start_time, record_thread
    if key == keyboard.Key.end:
        if not is_recording:
            start_time = time.time()
            record_thread = threading.Thread(target=start_recording)
            record_thread.start()


def on_release(key):
    global record_thread
    if key == keyboard.Key.end:
        if is_recording:
            stop_recording()
            if record_thread.is_alive():
                record_thread.join()


def start_recording():
    global is_recording, audio_data
    if not is_recording:
        is_recording = True
        audio_data = np.array([])
        with sd.InputStream(callback=audio_callback, samplerate=fs, channels=1):
            while is_recording:
                time.sleep(0.1)


def stop_recording():
    global is_recording, audio_data, start_time
    is_recording = False
    duration = len(audio_data) / fs
    if duration >= min_recording_duration:
        transcription = transcribe_audio(audio_data)
        if transcription:
            # Remove \n from the end if auto_enter is False
            if not auto_enter and transcription.endswith("\n"):
                transcription = transcription[
                    :-1
                ]  # Removes the last character if it's \n

            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Transcription: {transcription}")
            pyautogui.typewrite(transcription)


def audio_callback(indata, frames, time, status):
    global audio_data
    if status:
        print(status)
    audio_data = np.concatenate((audio_data, indata[:, 0]))


def transcribe_audio(audio):
    # Convert the NumPy array to a WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        write_wav(tmpfile.name, fs, audio)
        # Open the temporary file for reading and send it to OpenAI for transcription
        with open(tmpfile.name, "rb") as f:
            try:
                response = openai.audio.transcriptions.create(
                    model="whisper-1", file=f, response_format="text"
                )
                return response
            except Exception as e:
                print("Failed to transcribe audio: ", str(e))
    os.remove(tmpfile.name)  # Clean up the temporary file


def write_wav(file_path, samplerate, data):
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((data * 32767).astype(np.int16))


def main():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
