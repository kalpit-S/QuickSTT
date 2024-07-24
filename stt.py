import sounddevice as sd
from pynput import keyboard
import numpy as np
import wave
import time
import threading
import tempfile
import os
import pyautogui
import pyperclip
import re
from groq import Groq
from openai import OpenAI
import configparser
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration loading
config = configparser.ConfigParser()
config.read("config.ini")
groq_api_key = config.get("Groq", "api_key", fallback=None)
openai_api_key = config.get("OpenAI", "api_key", fallback=None)
auto_enter = config.getboolean("Settings", "auto_enter")
start_key = config.get("Hotkeys", "start_recording", fallback="f14")
use_llm_key = config.get("Hotkeys", "use_llm", fallback="f15")

logging.info("Configuration loaded.")

# Initialize client based on available API key
client_groq = None
client_openai = None

if groq_api_key:
    client_groq = Groq(api_key=groq_api_key)
    logging.info("Groq client initialized.")
elif openai_api_key:
    client_openai = OpenAI(api_key=openai_api_key)
    logging.info("OpenAI client initialized.")

# Global variables to manage recording state
is_recording = False
audio_data = np.array([])
start_time = 0
record_thread = None
fs = 44100  # Sample rate
min_recording_duration = 0.25  # Minimum recording duration (in seconds)
highlighted_text = ""
backspace_done = False  # Flag to ensure backspace is only pressed once


def start_recording():
    global is_recording, audio_data, start_time
    if not is_recording:
        logging.info("Starting recording...")
        is_recording = True
        audio_data = np.array([])
        start_time = time.time()
        with sd.InputStream(callback=audio_callback, samplerate=fs, channels=1):
            while is_recording:
                time.sleep(0.1)
        logging.info("Recording started.")


def stop_recording():
    global is_recording, audio_data
    logging.info("Stopping recording...")
    is_recording = False
    duration = len(audio_data) / fs
    if duration >= min_recording_duration:
        transcription = transcribe_audio(audio_data)
        if transcription:
            pyperclip.copy(transcription)
            pyautogui.hotkey("ctrl", "v")
            if auto_enter:
                pyautogui.hotkey("enter")
            logging.info("Transcription completed and pasted.")
    else:
        logging.info("Recording duration too short, no transcription done.")


def stop_recording_llm():
    global is_recording, audio_data, start_time, highlighted_text
    logging.info("Stopping LLM recording...")
    is_recording = False
    duration = len(audio_data) / fs
    if duration >= min_recording_duration:
        transcription = transcribe_audio(audio_data)
        if transcription:
            combined_text = f"""You are an assistant. Please process the following input and provide the appropriate response based on the given prompt. Only provide the response; do not add any commentary. If there is no input text, just respond to the prompt.

Input text: {highlighted_text}

Prompt: {transcription}

Output:"""
            response = generate_response_no_conversation(combined_text)
            if response:
                pyperclip.copy(response)
                pyautogui.hotkey("ctrl", "v")
                logging.info("LLM response generated and pasted.")
    else:
        logging.info("Recording duration too short, no transcription done.")


def audio_callback(indata, frames, time, status):
    global audio_data
    if status:
        logging.warning(f"Audio callback status: {status}")
    audio_data = np.concatenate((audio_data, indata[:, 0]))


def transcribe_audio(audio):
    logging.info("Transcribing audio...")
    # Convert the NumPy array to a WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        write_wav(tmpfile.name, fs, audio)
        # Open the temporary file for reading and send it to the API for transcription
        with open(tmpfile.name, "rb") as f:
            try:
                if client_groq:
                    transcription = client_groq.audio.transcriptions.create(
                        file=(tmpfile.name, f.read()),
                        model="whisper-large-v3",
                        response_format="text",
                        language="en",
                        temperature=0.0,
                    )
                    logging.info("Transcription from Groq completed.")
                    return transcription.text
                elif client_openai:
                    transcription = client_openai.audio.transcriptions.create(
                        file=f,
                        model="whisper-1",
                        response_format="text",
                        language="en",
                        temperature=0.0,
                    )
                    logging.info("Transcription from OpenAI completed.")
                    return transcription.text
            except Exception as e:
                logging.error(f"Failed to transcribe audio: {str(e)}")
    os.remove(tmpfile.name)  # Clean up the temporary file
    logging.info("Temporary audio file removed.")


def write_wav(file_path, samplerate, data):
    logging.info(f"Writing WAV file to {file_path}...")
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((data * 32767).astype(np.int16))
    logging.info("WAV file written.")


def generate_response_no_conversation(text):
    """Generate a response from the Groq or OpenAI chat model API based on the given text."""
    logging.info("Generating response from LLM...")
    try:
        if client_groq:
            response = client_groq.chat.completions.create(
                messages=[{"role": "user", "content": text}],
                model="llama-3.1-70b-versatile",
            )
            if response.choices[0].message.content:
                latest_message = response.choices[0].message.content
                logging.info("Response from Groq obtained.")
                return latest_message.strip()
            else:
                logging.info("No content in the Groq response.")
                return None
        elif client_openai:
            response = client_openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": text}],
            )
            if response.choices[0].message.content:
                latest_message = response.choices[0].message.content
                logging.info("Response from OpenAI obtained.")
                return latest_message.strip()
            else:
                logging.info("No content in the OpenAI response.")
                return None
    except Exception as e:
        logging.error(f"Failed to generate response: {str(e)}")
        return None


def start_llm_recording():
    global start_time, record_thread, highlighted_text, backspace_done, is_recording
    logging.info("Starting LLM recording...")
    if not is_recording:
        is_recording = True
        start_time = time.time()
        record_thread = threading.Thread(target=start_recording)
        record_thread.start()
        if not backspace_done:
            pyautogui.hotkey("ctrl", "c")
            time.sleep(0.1)
            highlighted_text = pyperclip.paste().strip()
            pyautogui.hotkey("backspace")
            backspace_done = True
            logging.info("Text highlighted and copied, backspace pressed.")


def stop_llm_recording():
    global backspace_done, is_recording
    logging.info("Stopping LLM recording...")
    if is_recording:
        stop_recording_llm()
    if record_thread and record_thread.is_alive():
        record_thread.join()
    backspace_done = False
    is_recording = False
    logging.info("LLM recording stopped.")


def on_press(key):
    # logging.debug(f"Key pressed: {key}")
    try:
        if key == getattr(keyboard.Key, start_key):
            start_recording()
    except AttributeError:
        pass


def on_release(key):
    logging.debug(f"Key released: {key}")
    try:
        if key == getattr(keyboard.Key, start_key):
            stop_recording()
        elif key == getattr(keyboard.Key, use_llm_key):
            stop_llm_recording()
    except AttributeError:
        pass


def main():
    logging.info("Starting keyboard listener...")
    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release,
    ) as listener:
        listener.join()


if __name__ == "__main__":
    main()
