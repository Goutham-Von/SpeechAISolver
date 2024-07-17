from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import sounddevice as sd
import json
import ollamahandler as ollama
import inquirer
from googleTranscriber import Transcriptor
import torch.multiprocessing as mp
import time

app = Flask(__name__)
socketio = SocketIO(app)


output = ""
recorded_text = ""
is_recording = False

def chat():
    global recorded_text, output, transcriptor, is_recording
    print("[+] Transcribing...")
    while True :
        text = transcriptor.getTranscription()
        if text is None :
            break
        recorded_text += text
    print("[+] Generating...")
    response = ollama.process(recorded_text)
    for line in response.iter_lines():
        if line:
            if(is_recording) :
                response.close()
                break
            data = json.loads(line)['message']['content']
            print(data, end = "")
            output += data


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
    global transcriptor, recorded_text
    is_recording = True
    recorded_text = ""
    transcriptor.start()

@socketio.on('stop_recording')
def stop_recording():
    global recorded_text, output
    is_recording = False 
    transcriptor.stop()
    output = ""
    chat()

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
    ollama.loadModel()
    transcriptor = Transcriptor(audioPrompt())
    socketio.run(app, host="0.0.0.0", port=9900)
    