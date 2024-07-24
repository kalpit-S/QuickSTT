import numpy as np
import sounddevice as sd
import time
import tempfile
import wave
import os
import threading
import pyautogui
import pyperclip
import re
from pynput import keyboard
from groq import Groq
from openai import OpenAI
import configparser
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize config parser
config = configparser.ConfigParser()
config_file_path = "config.ini"

# Read the config file
if os.path.exists(config_file_path):
    config.read(config_file_path)
    logging.info("Configuration file loaded.")
else:
    logging.error("Configuration file not found.")
    exit(1)

# Extract configurations
groq_api_key = config.get("Groq", "api_key", fallback=None)
openai_api_key = config.get("OpenAI", "api_key", fallback=None)
auto_enter = config.getboolean("Settings", "auto_enter", fallback=False)
clean_transcription_using_llm = config.getboolean(
    "Settings", "clean_transcription_using_llm", fallback=False
)
start_key = config.get("Hotkeys", "start_recording", fallback="f14")
use_llm_key = config.get("Hotkeys", "use_llm", fallback="f15")

# Initialize client
client = None
if groq_api_key:
    client = Groq(api_key=groq_api_key)
    logging.info("Groq client initialized.")
elif openai_api_key:
    client = OpenAI(api_key=openai_api_key)
    logging.info("OpenAI client initialized.")
else:
    logging.error(
        "No valid API key found. Please set either Groq or OpenAI API key in config.ini."
    )
    exit(1)

# Global variables
is_recording = False
audio_data = np.array([])
start_time = 0
fs = 44100  # Sample rate
min_recording_duration = 0.20  # Minimum recording duration (in seconds)
record_thread = None
highlighted_text = ""
backspace_done = False


def audio_callback(indata, frames, time, status):
    global audio_data
    if status:
        print(status)
    audio_data = np.concatenate((audio_data, indata[:, 0]))


def start_recording():
    global is_recording, audio_data, start_time
    if not is_recording:
        is_recording = True
        audio_data = np.array([])
        start_time = time.time()
        with sd.InputStream(callback=audio_callback, samplerate=fs, channels=1):
            while is_recording:
                time.sleep(0.1)


def stop_recording():
    global is_recording, audio_data
    is_recording = False
    duration = len(audio_data) / fs
    if duration >= min_recording_duration:
        transcription = transcribe_audio(audio_data)
        if transcription:
            end_time = time.time()
            total_time = end_time - start_time
            words = transcription.split()
            words_per_minute = (len(words) / total_time) * 60
            print(
                f"{total_time:.2f} secs | {words_per_minute:.2f} WPM | {transcription}"
            )
            pyperclip.copy(transcription)
            pyautogui.hotkey("ctrl", "v")
            if auto_enter:
                pyautogui.press("enter")


def transcribe_audio(audio):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        write_wav(tmpfile.name, fs, audio)
        with open(tmpfile.name, "rb") as f:
            try:
                if isinstance(client, Groq):
                    transcription = client.audio.transcriptions.create(
                        file=(tmpfile.name, f.read()),
                        model="whisper-large-v3",
                        response_format="text",
                        language="en",
                        temperature=0.0,
                    )
                    print(f"Transcription from Groq: {transcription}")
                    if clean_transcription_using_llm:
                        transcription = (
                            client.chat.completions.create(
                                model="llama-3.1-8b-instant",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "please clean up the transcription, do your best to infer what the user said. Only respond with the cleaned up transcription, nothing else.",
                                    },
                                    {
                                        "role": "user",
                                        "content": f"Transcription: {transcription}",
                                    },
                                ],
                            )
                            .choices[0]
                            .message.content.strip()
                        )
                        print(f"Cleaned up transcription: {transcription}")
                elif isinstance(client, OpenAI):
                    transcription = client.audio.transcriptions.create(
                        file=f,
                        model="whisper-1",
                        response_format="text",
                        language="en",
                        temperature=0.0,
                    )
                    if clean_transcription_using_llm:
                        transcription = (
                            client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "please clean up the transcription, do your best to infer what the user said. Only respond with the cleaned up transcription, nothing else.",
                                    },
                                    {
                                        "role": "user",
                                        "content": f"Transcription: {transcription}",
                                    },
                                ],
                            )
                            .choices[0]
                            .message.content.strip()
                        )
                        print(f"Cleaned up transcription: {transcription}")
                logging.info("Transcription completed.")
                return transcription
            except Exception as e:
                logging.error(f"Failed to transcribe audio: {str(e)}")
    os.remove(tmpfile.name)
    return ""


def write_wav(file_path, samplerate, data):
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((data * 32767).astype(np.int16))


def start_transcription_recording():
    global start_time, record_thread
    start_time = time.time()
    record_thread = threading.Thread(target=start_recording)
    record_thread.start()


def stop_transcription_recording():
    global record_thread
    if is_recording:
        stop_recording()
    if record_thread and record_thread.is_alive():
        record_thread.join()


def start_llm_recording():
    global start_time, record_thread, highlighted_text, backspace_done
    if not backspace_done:
        previous_clipboard = pyperclip.paste()
        pyautogui.hotkey("ctrl", "c")
        time.sleep(0.1)
        new_clipboard = pyperclip.paste().strip()
        if new_clipboard != previous_clipboard:
            highlighted_text = new_clipboard
        else:
            highlighted_text = ""
        pyautogui.press("backspace")
        backspace_done = True
    start_transcription_recording()


def stop_llm_recording():
    global backspace_done, is_recording, audio_data
    is_recording = False
    duration = len(audio_data) / fs
    if duration >= min_recording_duration:
        transcription = transcribe_audio(audio_data)
        if transcription:
            end_time = time.time()
            total_time = end_time - start_time
            words = transcription.split()
            words_per_minute = (len(words) / total_time) * 60
            print(
                f"{total_time:.2f} secs | {words_per_minute:.2f} WPM | Transcription: {transcription}"
            )

            llm_response = generate_llm_response(highlighted_text, transcription)
            if llm_response:
                print(f"LLM Response: {llm_response}")
                pyperclip.copy(llm_response)
                pyautogui.hotkey("ctrl", "v")
                if auto_enter:
                    pyautogui.press("enter")

    if record_thread and record_thread.is_alive():
        record_thread.join()
    backspace_done = False


def generate_llm_response(client, input_text, prompt):
    try:
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant responding to user prompts within their current application. Provide relevant, tailored responses that can be directly inserted into the user's active application.",
            },
            {
                "role": "user",
                "content": f"Highlighted text:\n{input_text}\n\nUser prompt:\n{prompt}\n\nRespond to the prompt, using highlighted text as context if provided. Give a direct response without commentary. Address only the prompt if no highlighted text is given. Your response will be inserted into the user's application. Avoid formatting elements like backticks or quotes.",
            },
        ]
        if isinstance(client, Groq):
            response = client.chat.completions.create(
                messages=messages,
                model="llama-3.1-70b-versatile",
            )
        elif isinstance(client, OpenAI):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
        else:
            raise ValueError("Unsupported client type")

        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Failed to generate LLM response: {str(e)}")
        return ""


def on_press(key):
    if key == getattr(keyboard.Key, start_key):
        start_transcription_recording()
    elif key == getattr(keyboard.Key, use_llm_key):
        start_llm_recording()


def on_release(key):
    if key == getattr(keyboard.Key, start_key):
        stop_transcription_recording()
    elif key == getattr(keyboard.Key, use_llm_key):
        stop_llm_recording()


def main():
    print(
        f"Press and hold {start_key} for transcription, {use_llm_key} for LLM processing."
    )
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
