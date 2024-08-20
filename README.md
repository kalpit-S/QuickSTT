# Voice Transcription and LLM Integration Tool

## Overview

This is a Python-based tool that allows you to record audio, transcribe it into text, and optionally clean up the transcription using a Large Language Model (LLM) like Groq or OpenAI's API. The tool also integrates with your clipboard, allowing you to paste the transcription directly into your current application, with the option to trigger an LLM response based on highlighted text.

## Features

- **Audio Recording and Transcription**: Record audio from your microphone and convert it into text using either the Groq API or OpenAI API.
- **LLM-Based Transcription Cleanup**: Optionally clean up the transcription using an LLM to improve readability and accuracy.
- **Hotkey Integration**: Use customizable hotkeys to start and stop audio recording, as well as trigger LLM processing.
- **Cross-Platform Compatibility**: Works on both Windows and macOS, with platform-specific adjustments for key mappings.
- **Clipboard Integration**: Automatically copies the transcription to the clipboard and pastes it into the current application.

## Requirements

- Python 3.x
- `numpy`
- `sounddevice`
- `pyautogui`
- `pyperclip`
- `pynput`
- `groq` (if using Groq API)
- `openai` (if using OpenAI API)

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-repo/voice-transcription-tool.git
   cd voice-transcription-tool
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration**:
   Create a `config.ini` file in the root directory of the project and fill in your API keys and settings as needed. You only need to provide one API key, either for Groq or OpenAI. Here's an example configuration:

   ```ini
   [Groq]
   api_key = your_groq_api_key_here

   [OpenAI]
   api_key = your_openai_api_key_here

   [Settings]
   auto_enter = true
   clean_transcription_using_llm = false

   [Hotkeys]
   start_recording = f15
   use_llm = f14
   ```

   **Note**: If you have both API keys, the tool will prioritize using the Groq API.

## Usage

1. **Run the Script**:

   ```bash
   python main.py
   ```

2. **Recording and Transcription**:

   - Press and hold the `start_recording` hotkey (default: `F15`) to start recording.
   - Release the hotkey to stop recording and transcribe the audio.

3. **LLM Processing**:

   - Highlight some text in your application.
   - Press and hold the `use_llm` hotkey (default: `F14`) to start recording and trigger the LLM processing.
   - The LLM will generate a response based on the highlighted text and transcription.

4. **Automatic Pasting**:
   - After transcription, the text will be copied to your clipboard and automatically pasted into your active application. If `auto_enter` is enabled, it will also press `Enter` to submit the text.

## Platform-Specific Notes

- **macOS**:

  - The tool automatically adjusts key mappings and clipboard shortcuts for macOS.
  - Ensure the application has microphone and accessibility permissions in System Preferences.

- **Windows**:
  - No additional setup is required.

## Troubleshooting

- **Configuration File Not Found**: Ensure that `config.ini` is in the root directory of your project and is correctly formatted.
- **Microphone Access**: On macOS, you may need to manually grant microphone access in System Preferences.
- **Hotkey Issues**: If the hotkeys don't work as expected, ensure the correct key names are used in the `config.ini` file, and adjust for your platform if necessary.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## Contact

For any issues or suggestions, feel free to open an issue on the GitHub repository or contact the maintainer at your.email@example.com.

---

This README provides the necessary information to get started with the tool, including installation instructions, usage guidance, and platform-specific notes. It also covers the requirement that only one API key (either Groq or OpenAI) is necessary.
