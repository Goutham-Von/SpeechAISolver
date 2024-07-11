from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import sounddevice as sd
import json
from threading import Thread
import ollamahandler as ollama
import threading
from MockAudioRecorder import resetAudioStream, stopAudioStream
from RealtimeSTT import AudioToTextRecorder
import inquirer
from loggingRtSTT import log

app = Flask(__name__)
socketio = SocketIO(app)


is_recording = False
recorded_text = ""
output = ""
recorder = None
recorder_ready = threading.Event()
audio_input_index = 0
whisper_speech_to_text_model = None


# Audio Processing
def onRecordStart() :
    global recorded_text, output
    recorded_text = ""

def onRecordStop() :
    global is_recording, output, recorded_text
    recorded_text = recorder.text()
    if(recorded_text != "") :
        print(f"Question : {recorded_text}")
        chat()
    else :
        output = "skipped Question due to short length"

def recorder_thread():
    global recorder, recorder_ready
    print("Initializing RealtimeSTT...")
    recorder = AudioToTextRecorder(
        model=whisper_speech_to_text_model, 
        input_device_index = audio_input_index,
        on_recording_start=onRecordStart,
        on_recording_stop=onRecordStop,
        min_gap_between_recordings=3,
        language='en',
        silero_sensitivity= 0.4,
        webrtc_sensitivity=2,
        post_speech_silence_duration= 0.2
    )
    print("RealtimeSTT initialized")
    recorder_ready.set()







ollama.loadModel()
def chat():
    global output, recorded_text, is_recording
    print("[+] Analysis...")
    response = ollama.process(recorded_text)
    output = ""
    for line in response.iter_lines():
        if line:
            if(is_recording) :
                response.close()
                break
            data = json.loads(line)['message']['content']
            print(data, end = "")
            output += data
    print("")


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
    global is_recording
    print("\n\nListening...")
    is_recording = True
    resetAudioStream(recorder)
    recorder.start()

@socketio.on('stop_recording')
def stop_recording():
    global is_recording
    is_recording = False
    stopAudioStream(recorder)
    recorder.stop()

def audioPrompt() :
    global audio_input_index
    audio_device_info = sd.query_devices()
    audio_device_prompt = [
        inquirer.List(
            'option',
            message="Choose an input audio device",
            choices=[f"{device['index']} {device['name']}" for device in audio_device_info],
        ),
    ]
    audioDevice = inquirer.prompt(audio_device_prompt)['option']
    audio_input_index = int(audioDevice.split(' ')[0])

def whisperSizePrompt() :
    global whisper_speech_to_text_model
    modelPrompt = [
        inquirer.List(
            'option',
            message="Choose Whisper Model Size",
            choices=['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'],
            default='small'
        )
    ]
    whisper_speech_to_text_model = inquirer.prompt(modelPrompt)['option']
    

if __name__ == '__main__':
    # # Register the signal handlers
    # signal.signal(signal.SIGTERM, signal_handler) # also signal.SIGINT
    audioPrompt()
    whisperSizePrompt()

    socketio.start_background_task(target=recorder_thread)
    socketio.run(app, host="0.0.0.0", port=9900)