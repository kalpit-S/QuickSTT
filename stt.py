import numpy as np
import sounddevice as sd
import time
import tempfile
import wave
import os
import threading
import pyautogui
import pyperclip
import platform
import logging
import configparser
from pynput import keyboard
from groq import Groq
from openai import OpenAI

# Constants
FS = 44100  # Sample rate
MIN_RECORDING_DURATION = 0.20  # Minimum recording duration in seconds

# Initialize logging
logging.basicConfig(level=logging.INFO)


# Load configurations
def load_config():
    config = configparser.ConfigParser()
    config_file_path = "config.ini"

    if not os.path.exists(config_file_path):
        logging.error("Configuration file not found.")
        exit(1)

    config.read(config_file_path)
    logging.info("Configuration file loaded.")
    return config


config = load_config()

# Extract configurations
groq_api_key = config.get("Groq", "api_key", fallback=None)
openai_api_key = config.get("OpenAI", "api_key", fallback=None)
auto_enter = config.getboolean("Settings", "auto_enter", fallback=False)
clean_transcription_using_llm = config.getboolean(
    "Settings", "clean_transcription_using_llm", fallback=False
)
start_key = config.get("Hotkeys", "start_recording", fallback="f14")
use_llm_key = config.get("Hotkeys", "use_llm", fallback="f15")
copy_paste_key = "cmd" if platform.system() == "Darwin" else "ctrl"


# Initialize client
def initialize_client(groq_api_key, openai_api_key):
    if groq_api_key:
        logging.info("Groq client initialized.")
        return Groq(api_key=groq_api_key)
    elif openai_api_key:
        logging.info("OpenAI client initialized.")
        return OpenAI(api_key=openai_api_key)
    else:
        logging.error(
            "No valid API key found. Please set either Groq or OpenAI API key in config.ini."
        )
        exit(1)


client = initialize_client(groq_api_key, openai_api_key)

# Global variables
is_recording = False
audio_data = np.array([])
start_time = 0
record_thread = None
highlighted_text = ""
backspace_done = False


def audio_callback(indata, frames, time, status):
    global audio_data
    if status:
        logging.warning(f"Audio status: {status}")
    audio_data = np.concatenate((audio_data, indata[:, 0]))


def start_recording():
    global is_recording, audio_data, start_time
    if not is_recording:
        is_recording = True
        audio_data = np.array([])
        start_time = time.time()
        logging.info("Recording started.")
        with sd.InputStream(callback=audio_callback, samplerate=FS, channels=1):
            while is_recording:
                time.sleep(0.1)


def stop_recording():
    global is_recording, audio_data
    is_recording = False
    duration = len(audio_data) / FS
    if duration >= MIN_RECORDING_DURATION:
        logging.info(f"Recording stopped. Duration: {duration:.2f} seconds.")
        transcription = transcribe_audio(audio_data)
        if transcription:
            handle_transcription(transcription)


def transcribe_audio(audio):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        write_wav(tmpfile.name, FS, audio)
        tmpfile_name = tmpfile.name  # Save the filename to delete it later

    try:
        with open(tmpfile_name, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=f,
                model=(
                    "whisper-1" if isinstance(client, OpenAI) else "whisper-large-v3"
                ),
                response_format="text",
                language="en",
                temperature=0.0,
            )
            logging.info(f"Transcription: {transcription}")
            if clean_transcription_using_llm:
                transcription = clean_transcription(transcription)
            return transcription
    except Exception as e:
        logging.error(f"Failed to transcribe audio: {str(e)}")
    finally:
        try:
            os.remove(tmpfile_name)
        except PermissionError as e:
            logging.error(f"Could not delete temporary file: {e}")

    return ""


def clean_transcription(transcription):
    try:
        response = client.chat.completions.create(
            model=("gpt-4o-mini" if isinstance(client, OpenAI) else "llama3-70b-8192"),
            messages=[
                {
                    "role": "system",
                    "content": "Clean up the whisper transcription, inferring the user's intent while staying faithful to the original text.",
                },
                {"role": "user", "content": f"Transcription: {transcription}"},
            ],
        )
        cleaned_transcription = response.choices[0].message.content.strip()
        logging.info(f"Cleaned transcription: {cleaned_transcription}")
        return cleaned_transcription
    except Exception as e:
        logging.error(f"Failed to clean transcription: {str(e)}")
        return transcription


def write_wav(file_path, samplerate, data):
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((data * 32767).astype(np.int16))


def handle_transcription(transcription):
    end_time = time.time()
    total_time = end_time - start_time
    words = transcription.split()
    words_per_minute = (len(words) / total_time) * 60
    logging.info(f"Total time: {total_time:.2f} secs | WPM: {words_per_minute:.2f}")
    pyperclip.copy(transcription)
    pyautogui.hotkey(copy_paste_key, "v")
    if auto_enter:
        pyautogui.press("enter")


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
        highlighted_text = get_highlighted_text()
        pyautogui.press("backspace")
        backspace_done = True
    start_transcription_recording()


def stop_llm_recording():
    global backspace_done, is_recording, audio_data
    is_recording = False
    duration = len(audio_data) / FS
    if duration >= MIN_RECORDING_DURATION:
        transcription = transcribe_audio(audio_data)
        if transcription:
            llm_response = generate_llm_response(highlighted_text, transcription)
            if llm_response:
                pyperclip.copy(llm_response)
                pyautogui.hotkey(copy_paste_key, "v")
    if record_thread and record_thread.is_alive():
        record_thread.join()
    backspace_done = False


def get_highlighted_text():
    previous_clipboard = pyperclip.paste()
    pyautogui.hotkey(copy_paste_key, "c")
    time.sleep(0.1)
    new_clipboard = pyperclip.paste().strip()
    return new_clipboard if new_clipboard != previous_clipboard else ""


conversation_history = []


def generate_llm_response(highlighted_text, transcription):
    global conversation_history

    try:
        # Prepare the new message, handling empty inputs
        user_content = ""
        if highlighted_text:
            user_content += f"Highlighted text: {highlighted_text}\n\n"
        if transcription:
            user_content += f"User prompt: {transcription}"

        if not user_content:
            user_content = "No input provided."

        new_user_message = {"role": "user", "content": user_content.strip()}

        # Add the new message to the conversation history
        conversation_history.append(new_user_message)

        # Keep only the last 10 messages to maintain context without overwhelming the model
        conversation_history = conversation_history[-10:]

        # Prepare the messages for the API call
        messages = (
            [
                {
                    "role": "system",
                    "content": """You are an AI assistant that helps users modify text directly within their current application. Your responses should be the modified text that can be directly pasted into the user's document. Follow these guidelines:

1. If given highlighted text and a task, modify the highlighted text according to the task.
2. If only given a task without highlighted text, provide a brief, direct response to the task.
3. For code-related tasks, include the entire modified code, not just the changes.
4. For text improvement tasks (grammar, spelling, etc.), provide the full corrected text.
5. Do not use backticks, quotes, or any other formatting in your response.
6. Do not include any explanations, comments, or anything other than the modified text or direct answer.
7. If the task is unclear, ask for clarification in a concise manner.
""",
                }
            ]
            + conversation_history
        )

        # logging.info(f"Sending the following messages to the LLM: {messages}")

        response = client.chat.completions.create(
            messages=messages,
            model=(
                "gpt-4o-mini"
                if isinstance(client, OpenAI)
                else "llama-3.1-70b-versatile"
            ),
            temperature=0.7,
            max_tokens=150,
        )

        assistant_response = response.choices[0].message.content.strip()

        # Add the assistant's response to the conversation history
        conversation_history.append(
            {"role": "assistant", "content": assistant_response}
        )

        # logging.info(f"Updated conversation history: {conversation_history}")

        return assistant_response
    except Exception as e:
        logging.error(f"Failed to generate LLM response: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again."


def on_press(key):
    if key == keyboard.Key[start_key]:
        start_transcription_recording()
    elif key == keyboard.Key[use_llm_key]:
        start_llm_recording()


def on_release(key):
    if key == keyboard.Key[start_key]:
        stop_transcription_recording()
    elif key == keyboard.Key[use_llm_key]:
        stop_llm_recording()


def main():
    logging.info(
        f"Press and hold {start_key} for transcription, {use_llm_key} for LLM processing."
    )
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
