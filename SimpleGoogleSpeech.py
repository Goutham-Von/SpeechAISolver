from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import sounddevice as sd
import json
import ollamahandler as ollama
import inquirer
import speech_recognition as sr
import pyaudio
import threading
import logging
import argparse

class AudioRecording :

    FILENAME = 'google-speech.wav'
    CHUNK = 512  # Buffer size
    PYAUDIO_FORMAT = pyaudio.paInt16  # 16-bit resolution
    CHANNELS = 1  # Mono
    RATE = 16000  # 16kHz sample rate

    def __init__(self, input_device_index = 1) :
        self.audioInterface = pyaudio.PyAudio()
        self.recognizer = sr.Recognizer()
        self.input_device_index = input_device_index
        self.stream = None
        self.is_recording = threading.Event()
        self.recording_thread = None

    def audio_to_text(self):
        if debug:
            print('[+] transcribing...')
        audio_data = b''.join(self.frames)
        audio = sr.AudioData(audio_data, AudioRecording.RATE, 2)
        text = self.recognizer.recognize_google(audio)
        return text
        
    def stop(self):
        self.is_recording.clear()
        self.recording_thread.join()
        return self.audio_to_text()
        
    def start(self):
        self.stream = self.audioInterface.open(
                format=AudioRecording.PYAUDIO_FORMAT,
                channels=AudioRecording.CHANNELS,
                rate=AudioRecording.RATE,
                input=True,
                frames_per_buffer=AudioRecording.CHUNK,
                input_device_index=self.input_device_index
            )
        self.is_recording.set()
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()

    def record_audio(self):
        self.frames = []
        if debug :
            print('[+] recording...')
        while self.is_recording.is_set() :
            data = self.stream.read(AudioRecording.CHUNK)
            self.frames.append(data)

        self.stream.stop_stream()
        self.stream.close()

output = ""
recorded_text = ""
is_recording = threading.Event()
chat_thread = None

def chat():
    global recorded_text, output, is_recording, audioRecorder
    recorded_text = audioRecorder.stop()
    if debug :
        print(f'\n\nQuestion : {recorded_text}')
    response = ollama.process(recorded_text)
    for line in response.iter_lines():
        if line:
            if(is_recording.is_set()) :
                response.close()
                break
            data = json.loads(line)['message']['content']
            if debug :
                print(data, end = "")
            output += data


app = Flask(__name__)
socketio = SocketIO(app)

# Flask Routes and socketio
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcription', methods=['GET'])
def get_transcription():
    global output, recorded_text
    return jsonify({"answer": f"Question : {recorded_text}\n Solution : \n{output}"})

@socketio.on('start_recording')
def start_recording():
    global recorded_text
    is_recording.set()
    if chat_thread is not None :
        chat_thread.join()
    recorded_text = ""
    audioRecorder.start()

@socketio.on('stop_recording')
def stop_recording():
    global recorded_text, output
    is_recording.clear()
    output = ""
    chat_thread = threading.Thread(target=chat)
    chat_thread.start()



def audioPrompt() :
    audio_device_info = sd.query_devices()
    audio_device_prompt = [
        inquirer.List(
            'option',
            message="Choose an input audio device",
            choices=[f"{device['index']} {device['name']}" for device in audio_device_info],
        ),
    ]
    audioDevice = inquirer.prompt(audio_device_prompt)['option']
    return int(audioDevice.split(' ')[0])

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Flask application.')
    parser.add_argument('--debug', type=bool, default=True, help='Enable debug mode')
    debug = parser.parse_args().debug

    # Suppress Flask request logs
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    ollama.loadModel()
    audioRecorder = AudioRecording(audioPrompt())
    socketio.run(app, host="0.0.0.0", port=9900)
    