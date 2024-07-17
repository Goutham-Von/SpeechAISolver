import multiprocessing as mp
import pyaudio
import speech_recognition as sr
import logging
import time
    
logging.basicConfig(
    filename='transcriptor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Transcriptor:
    CHUNK = 512  # Buffer size
    PYAUDIO_FORMAT = pyaudio.paInt16  # 16-bit resolution
    CHANNELS = 1  # Mono
    RATE = 16000  # 16kHz sample rate

    def __init__(self, input_device_index=2):
        self.recognizer = sr.Recognizer()
        self.input_device_index = input_device_index

        try :
            self.audioInterface = pyaudio.PyAudio()
        except Exception as e:
            logging.log(logging.INFO, f"Error creating PyAudio object {e}")

        self.recording_state = mp.Event()
        self.recorder_process = None
        self.transcriber_process = None
        logging.log(logging.INFO, "Transcriptor initialized")
        # mp.set_start_method('swarm')

    def start(self):
        self.forceKillWorkerThreads()
        self.recording_state.set()
        self.audioQueue = mp.Queue()
        self.transcriptions = mp.Queue()

        logging.log(logging.INFO, "Recording started")
        print('\nListening...')
        self.recorder_process = mp.Process(
            target=Transcriptor._recorder_worker,
            args=(
                self.recording_state,
                self.audioQueue,
                self.audioInterface,
                self.input_device_index
            ))
        self.recorder_process.start()
        self.transcriber_process = mp.Process(
            target=Transcriptor._transcribe_worker,
            args=(
                self.recording_state,
                self.audioQueue,
                self.recognizer,
                self.transcriptions
            ))
        self.transcriber_process.start()

    def getTranscription(self):
        if(self.transcriptions.empty()) :
            return ""
        return self.transcriptions.get()

    @staticmethod
    def _recorder_worker(
        recording_state,
        audioQueue,
        audioInterface,
        input_device_index=1
    ):
        logging.log(logging.INFO, "Recording worker started")
        try :
            stream = audioInterface.open(
                format=Transcriptor.PYAUDIO_FORMAT,
                channels=Transcriptor.CHANNELS,
                rate=Transcriptor.RATE,
                input=True,
                frames_per_buffer=Transcriptor.CHUNK,
                input_device_index=input_device_index
            )
        except Exception as e:
            logging.log(logging.INFO, f"Error opening audio stream {e}")
        while recording_state.is_set():
            data = stream.read(Transcriptor.CHUNK, exception_on_overflow=False)
            audioQueue.put(data)
        logging.log(logging.INFO, "Recording worker stopped")

    @staticmethod
    def _transcribe_worker(recording_state, audioQueue, recognizer, transcriptions):
        logging.log(logging.INFO, "Transcribe worker started")
        frameDuration = int((Transcriptor.RATE / Transcriptor.CHUNK) * 4)
        frames = []
        transcription = ""
        while recording_state.is_set() or not audioQueue.empty():
            if(not audioQueue.empty()) :
                try:
                    data = audioQueue.get()
                    frames.append(data)
                    if len(frames) >= frameDuration:  # 4 seconds of audio data
                        audio_data = b''.join(frames)
                        audio = sr.AudioData(audio_data, Transcriptor.RATE, 2)
                        transcription = " " + recognizer.recognize_google(audio)
                        frames = []
                        transcriptions.put(transcription)
                        print(transcription)
                except Exception as e:
                    continue
            else :
                time.sleep(0.3)
        transcriptions.put(None)
        logging.log(logging.INFO, "Transcribe worker stopped")

    def forceKillWorkerThreads(self):
        logging.log(logging.INFO, "Killing worker threads")
        if self.recorder_process and self.recorder_process.is_alive():
            self.recorder_process.terminate()
        if self.transcriber_process and self.transcriber_process.is_alive():
            self.transcriber_process.terminate()

    def stop(self):
        logging.log(logging.INFO, "Stopping live transcription")
        self.recording_state.clear()
        self.recorder_process.join()
        if hasattr(self, 'stream') and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        self.transcriber_process.join()