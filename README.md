# Setup

Follow these steps to get the speech-to-text utility up and running:

1. **Install Python 3.6 or Later**: Ensure Python 3.6+ is installed on your system.
2. **Obtain an OpenAI API Key**: Sign up at OpenAI and obtain an API key. Edit `config.ini` in the project directory to include your API key:

   ```ini
   [OpenAI]
   api_key=your_api_key_here

   [Settings]
   auto_enter=true
   ```

3. **Install Required Dependencies**: Open your terminal or command prompt and run:

   ```bash
   pip install sounddevice pynput numpy pyautogui openai
   ```

# Usage

To use the utility:

- **Start**: Run `python stt.py` from within the project directory.
- **Activate**: Hold the `End` key to begin recording. Release it to stop and transcribe your speech to text. (Highly recommend binding to a mouse button)

# Configuration

- To toggle automatic submission of transcribed text, adjust the `auto_enter` setting in `config.ini` to `true` (for auto-submit) or `false` (requires manual submission).

# Customizing the Push-to-Talk Key

The default key for activating recording is `End`. To change it:

1. **Identify the New Key**: Consult the [pynput keyboard documentation](https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key) for key names.
2. **Edit the Script**: Open `speech_to_text.py` in a text editor.
3. **Find the Key Check**: In `on_press` and `on_release` functions, locate `keyboard.Key.end`.
4. **Replace the Key**: Change `keyboard.Key.end` to your chosen key, e.g., `keyboard.Key.f12` for the F12 key.

   ```python
   def on_press(key):
       global start_time, record_thread
       if key == keyboard.Key.f12:  # Change to your chosen key
           # Function code remains unchanged

   def on_release(key):
       global record_thread
       if key == keyboard.Key.f12:  # Change to your chosen key
           # Function code remains unchanged
   ```

5. **Save Changes**: After editing, save the file and rerun the script to test the new key configuration.
