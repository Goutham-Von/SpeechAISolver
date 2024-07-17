from typing import List, Optional, Union
import torch.multiprocessing as mp
import torch
from typing import List, Union
from ctypes import c_bool
from scipy.signal import resample
from scipy import signal
import faster_whisper
import collections
import numpy as np
import threading
import webrtcvad
import pyaudio
import logging
import halo
import time
import copy
import os
import re
import gc
import sounddevice as sd
import inquirer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

INIT_MODEL_TRANSCRIPTION = "small"
INIT_MODEL_TRANSCRIPTION_REALTIME = "small"
INIT_REALTIME_PROCESSING_PAUSE = 0.2
INIT_SILERO_SENSITIVITY = 0.4
INIT_WEBRTC_SENSITIVITY = 3
INIT_POST_SPEECH_SILENCE_DURATION = 0.6
INIT_MIN_LENGTH_OF_RECORDING = 0.5
INIT_MIN_GAP_BETWEEN_RECORDINGS = 0
ALLOWED_LATENCY_LIMIT = 10

TIME_SLEEP = 0.02
SAMPLE_RATE = 16000
BUFFER_SIZE = 512
INT16_MAX_ABS_VALUE = 32768.0



class AudioWhisper:
    def __init__(self,
                 model: str = INIT_MODEL_TRANSCRIPTION,
                 language: str = "en",
                 compute_type: str = "default",
                 input_device_index: int = None,
                 gpu_device_index: Union[int, List[int]] = 0,
                 device: str = "cpu",
                 use_microphone=True,
                 spinner=True,

                 # Realtime transcription parameters
                 enable_realtime_transcription=False,
                 on_realtime_transcription_update=None,

                 # Voice activation parameters
                 silero_sensitivity: float = INIT_SILERO_SENSITIVITY,
                 webrtc_sensitivity: int = INIT_WEBRTC_SENSITIVITY,
                 post_speech_silence_duration: float = (
                     INIT_POST_SPEECH_SILENCE_DURATION
                 ),
                 min_length_of_recording: float = (
                     INIT_MIN_LENGTH_OF_RECORDING
                 ),
                 min_gap_between_recordings: float = (
                     INIT_MIN_GAP_BETWEEN_RECORDINGS
                 ),
                 beam_size: int = 5,
                 beam_size_realtime: int = 3,
                 buffer_size: int = BUFFER_SIZE,
                 sample_rate: int = SAMPLE_RATE,
                 suppress_tokens: Optional[List[int]] = [-1],
                 ):
        self.language = language
        self.compute_type = compute_type
        self.input_device_index = input_device_index
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.use_microphone = mp.Value(c_bool, use_microphone)
        self.min_gap_between_recordings = min_gap_between_recordings
        self.min_length_of_recording = min_length_of_recording
        self.post_speech_silence_duration = post_speech_silence_duration
        self.enable_realtime_transcription = enable_realtime_transcription
        self.on_realtime_transcription_update = (
            on_realtime_transcription_update
        )
        self.beam_size = beam_size
        self.beam_size_realtime = beam_size_realtime
        self.allowed_latency_limit = ALLOWED_LATENCY_LIMIT

        self.audio_queue = mp.Queue()
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.silero_check_time = 0
        self.silero_working = False
        self.speech_end_silence_start = 0
        self.silero_sensitivity = silero_sensitivity
        self.listen_start = 0
        self.spinner = spinner
        self.halo = None
        self.state = "inactive"
        self.text_storage = []
        self.realtime_stabilized_safetext = ""
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False
        self.recording_thread = None
        self.realtime_thread = None
        self.audio_interface = None
        self.audio = None
        self.start_recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.last_transcription_bytes = None
        self.suppress_tokens = suppress_tokens

        self.is_shut_down = False
        self.shutdown_event = mp.Event()

        logging.info("Starting RealTimeSTT")

        self.interrupt_stop_event = mp.Event()
        self.was_interrupted = mp.Event()
        self.main_transcription_ready_event = mp.Event()
        self.parent_transcription_pipe, child_transcription_pipe = mp.Pipe()

        # Set device for model
        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"

        self.transcript_process = mp.Process(
            target=AudioWhisper._transcription_worker,
            args=(
                child_transcription_pipe,
                model,
                self.compute_type,
                self.gpu_device_index,
                self.device,
                self.main_transcription_ready_event,
                self.shutdown_event,
                self.beam_size,
                self.suppress_tokens
            )
        )
        self.transcript_process.start()
        self.realtime_model_type = model

        # Start audio data reading process
        if self.use_microphone.value:
            self.reader_process = mp.Process(
                target=AudioWhisper._audio_data_worker,
                args=(
                    self.audio_queue,
                    self.sample_rate,
                    self.buffer_size,
                    self.input_device_index,
                    self.shutdown_event,
                    self.use_microphone
                )
            )
            self.reader_process.start()

        # Initialize the realtime transcription model
        if self.enable_realtime_transcription:
            self.realtime_model_type = faster_whisper.WhisperModel(
                model_size_or_path=self.realtime_model_type,
                device=self.device,
                compute_type=self.compute_type,
                device_index=self.gpu_device_index
            )

        self.webrtc_vad_model = webrtcvad.Vad()
        self.webrtc_vad_model.set_mode(webrtc_sensitivity)

        # Setup voice activity detection model Silero VAD
        self.silero_vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            verbose=False,
            onnx=False
        )

        self.audio_buffer = collections.deque(
            maxlen=int(self.sample_rate // self.buffer_size)
        )
        self.frames = []

        # Recording control flags
        self.is_recording = False
        self.is_running = True
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False

        # Start the recording worker thread
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        # Start the realtime transcription worker thread
        self.realtime_thread = threading.Thread(target=self._realtime_worker)
        self.realtime_thread.daemon = True
        self.realtime_thread.start()

        # Wait for transcription models to start
        self.main_transcription_ready_event.wait()

        logging.debug('RealtimeSTT initialization completed successfully')

    @staticmethod
    def _transcription_worker(conn,
                              model_path,
                              compute_type,
                              gpu_device_index,
                              device,
                              ready_event,
                              shutdown_event,
                              beam_size,
                              suppress_tokens
                              ):
        model = faster_whisper.WhisperModel(
            model_size_or_path=model_path,
            device=device,
            compute_type=compute_type,
            device_index=gpu_device_index,
        )

        ready_event.set()

        while not shutdown_event.is_set():
            if conn.poll(0.5):
                audio, language = conn.recv()
                try:
                    segments = model.transcribe(
                        audio,
                        language=language if language else None,
                        beam_size=beam_size,
                        suppress_tokens=suppress_tokens
                    )
                    segments = segments[0]
                    transcription = " ".join(seg.text for seg in segments)
                    transcription = transcription.strip()
                    conn.send(('success', transcription))
                except Exception as e:
                    logging.error(f"General transcription error: {e}")
                    conn.send(('error', str(e)))
            else:
                time.sleep(0.02)

    @staticmethod
    def _audio_data_worker(audio_queue,
                           sample_rate,
                           buffer_size,
                           input_device_index,
                           shutdown_event,
                           use_microphone):
        audio_interface = pyaudio.PyAudio()
        if input_device_index is None:
            default_device = audio_interface.get_default_input_device_info()
            input_device_index = default_device['index']
        stream = audio_interface.open(
            rate=sample_rate,
            format=pyaudio.paInt16,
            channels=1,
            input=True,
            frames_per_buffer=buffer_size,
            input_device_index=input_device_index,
            )

        while not shutdown_event.is_set():
            data = stream.read(buffer_size)

            if use_microphone.value:
                audio_queue.put(data)

    def wakeup(self):
        self.listen_start = time.time()

    def abort(self):
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False
        self._set_state("inactive")
        self.interrupt_stop_event.set()
        self.was_interrupted.wait()
        self.was_interrupted.clear()

    def wait_audio(self):
        self.listen_start = time.time()

        # If not yet started recording, wait for voice activity to initiate.
        if not self.is_recording and not self.frames:
            self._set_state("listening")
            self.start_recording_on_voice_activity = True

            # Wait until recording starts
            while not self.interrupt_stop_event.is_set():
                if self.start_recording_event.wait(timeout=0.02):
                    break

        # If recording is ongoing, wait for voice inactivity
        # to finish recording.
        if self.is_recording:
            self.stop_recording_on_voice_deactivity = True

            # Wait until recording stops
            while not self.interrupt_stop_event.is_set():
                if (self.stop_recording_event.wait(timeout=0.02)):
                    break

        # Convert recorded frames to the appropriate audio format.
        audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)
        self.audio = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
        self.frames.clear()

        # Reset recording-related timestamps
        self.recording_stop_time = 0
        self.listen_start = 0

        self._set_state("inactive")

    def transcribe(self):
        self._set_state("transcribing")
        audio_copy = copy.deepcopy(self.audio)
        self.parent_transcription_pipe.send((self.audio, self.language))
        status, result = self.parent_transcription_pipe.recv()

        self._set_state("inactive")
        if status == 'success':
            self.last_transcription_bytes = audio_copy
            return self._preprocess_output(result)
        
    def text(self,
             on_transcription_finished=None,
             ):

        self.interrupt_stop_event.clear()
        self.was_interrupted.clear()

        self.wait_audio()

        if self.is_shut_down or self.interrupt_stop_event.is_set():
            if self.interrupt_stop_event.is_set():
                self.was_interrupted.set()
            return ""

        if on_transcription_finished:
            threading.Thread(target=on_transcription_finished,
                             args=(self.transcribe(),)).start()
        else:
            return self.transcribe()

    def start(self):

        # Ensure there's a minimum interval
        # between stopping and starting recording
        if (time.time() - self.recording_stop_time
                < self.min_gap_between_recordings):
            logging.info("Attempted to start recording "
                         "too soon after stopping."
                         )
            return self

        self._set_state("recording")
        self.text_storage = []
        self.realtime_stabilized_safetext = ""
        self.frames = []
        self.is_recording = True
        self.recording_start_time = time.time()
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.stop_recording_event.clear()
        self.start_recording_event.set()

        if self.on_recording_start:
            self.on_recording_start()

        return self

    def stop(self):

        # Ensure there's a minimum interval
        # between starting and stopping recording
        if (time.time() - self.recording_start_time
                < self.min_length_of_recording):
            return self

        logging.info("recording stopped")
        self.is_recording = False
        self.recording_stop_time = time.time()
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.silero_check_time = 0
        self.start_recording_event.clear()
        self.stop_recording_event.set()

        if self.on_recording_stop:
            self.on_recording_stop()

        return self

    def set_microphone(self, microphone_on=True):
        logging.info("Setting microphone to: " + str(microphone_on))
        self.use_microphone.value = microphone_on

    def shutdown(self):
        self.is_shut_down = True
        self.start_recording_event.set()
        self.stop_recording_event.set()

        self.shutdown_event.set()
        self.is_recording = False
        self.is_running = False

        logging.debug('Finishing recording thread')
        if self.recording_thread:
            self.recording_thread.join()

        logging.debug('Terminating reader process')

        # Give it some time to finish the loop and cleanup.
        if self.use_microphone:
            self.reader_process.join(timeout=10)

        if self.reader_process.is_alive():
            logging.warning("Reader process did not terminate "
                            "in time. Terminating forcefully."
                            )
            self.reader_process.terminate()

        logging.debug('Terminating transcription process')
        self.transcript_process.join(timeout=10)

        if self.transcript_process.is_alive():
            logging.warning("Transcript process did not terminate "
                            "in time. Terminating forcefully."
                            )
            self.transcript_process.terminate()

        self.parent_transcription_pipe.close()

        logging.debug('Finishing realtime thread')
        if self.realtime_thread:
            self.realtime_thread.join()

        if self.enable_realtime_transcription:
            if self.realtime_model_type:
                del self.realtime_model_type
                self.realtime_model_type = None
        gc.collect()

    def _recording_worker(self):
        logging.debug('Starting recording worker')

        try:
            was_recording = False

            # Continuously monitor audio for voice activity
            while self.is_running:

                try:

                    data = self.audio_queue.get()
                    if self.on_recorded_chunk:
                        self.on_recorded_chunk(data)
                except BrokenPipeError:
                    print("BrokenPipeError _recording_worker")
                    self.is_running = False
                    break

                if not self.is_recording:

                    # Set state and spinner text
                    if not self.recording_stop_time:
                        if self.listen_start:
                            self._set_state("listening")
                        else:
                            self._set_state("inactive")

                    # Check for voice activity to
                    # trigger the start of recording
                    self.speech_end_silence_start = 0

                else:
                    # If we are currently recording

                    # Stop the recording if silence is detected after speech
                    if self.stop_recording_on_voice_deactivity:

                        if not self._is_webrtc_speech(data, True):

                            # Voice deactivity was detected, so we start
                            # measuring silence time before stopping recording
                            if self.speech_end_silence_start == 0:
                                self.speech_end_silence_start = time.time()

                        else:
                            self.speech_end_silence_start = 0

                        # Wait for silence to stop recording after speech
                        if self.speech_end_silence_start and time.time() - \
                                self.speech_end_silence_start > \
                                self.post_speech_silence_duration:
                            logging.info("voice deactivity detected")
                            self.stop()

                if not self.is_recording and was_recording:
                    # Reset after stopping recording to ensure clean state
                    self.stop_recording_on_voice_deactivity = False

                if time.time() - self.silero_check_time > 0.1:
                    self.silero_check_time = 0

                was_recording = self.is_recording

                if self.is_recording:
                    self.frames.append(data)

                if not self.is_recording or self.speech_end_silence_start:
                    self.audio_buffer.append(data)

        except Exception as e:
            if not self.interrupt_stop_event.is_set():
                logging.error(f"Unhandled exeption in _recording_worker: {e}")
                raise

    def _realtime_worker(self):
        # Return immediately if real-time transcription is not enabled
        if not self.enable_realtime_transcription:
            return

        # Continue running as long as the main process is active
        while self.is_running:

            # Check if the recording is active
            if self.is_recording:

                # Sleep for the duration of the transcription resolution
                time.sleep(INIT_REALTIME_PROCESSING_PAUSE)

                # Convert the buffer frames to a NumPy array
                audio_array = np.frombuffer(
                    b''.join(self.frames),
                    dtype=np.int16
                    )

                # Normalize the array to a [-1, 1] range
                audio_array = audio_array.astype(np.float32) / \
                    INT16_MAX_ABS_VALUE

                # Perform transcription and assemble the text
                segments = self.realtime_model_type.transcribe(
                    audio_array,
                    language=self.language if self.language else None,
                    beam_size=self.beam_size_realtime,
                    initial_prompt=self.initial_prompt,
                    suppress_tokens=self.suppress_tokens,
                )

                # double check recording state
                # because it could have changed mid-transcription
                if self.is_recording and time.time() - \
                        self.recording_start_time > 0.5:

                    logging.debug('Starting realtime transcription')
                    self.realtime_transcription_text = " ".join(
                        seg.text for seg in segments[0]
                    )
                    self.realtime_transcription_text = \
                        self.realtime_transcription_text.strip()

                    self.text_storage.append(
                        self.realtime_transcription_text
                        )

                    # Take the last two texts in storage, if they exist
                    if len(self.text_storage) >= 2:
                        last_two_texts = self.text_storage[-2:]

                        # Find the longest common prefix
                        # between the two texts
                        prefix = os.path.commonprefix(
                            [last_two_texts[0], last_two_texts[1]]
                            )

                        # This prefix is the text that was transcripted
                        # two times in the same way
                        # Store as "safely detected text"
                        if len(prefix) >= \
                                len(self.realtime_stabilized_safetext):

                            # Only store when longer than the previous
                            # as additional security
                            self.realtime_stabilized_safetext = prefix

                    # Find parts of the stabilized text
                    # in the freshly transcripted text
                    matching_pos = self._find_tail_match_in_text(
                        self.realtime_stabilized_safetext,
                        self.realtime_transcription_text
                        )

                    if matching_pos >=0 :
                        # We found parts of the stabilized text
                        # in the transcripted text
                        # We now take the stabilized text
                        # and add only the freshly transcripted part to it
                        output_text = self.realtime_stabilized_safetext + \
                            self.realtime_transcription_text[matching_pos:]

                    # Invoke the callback with the transcribed text
                    self._on_realtime_transcription_update(
                        self._preprocess_output(
                            self.realtime_transcription_text,
                            True
                        )
                    )

            # If not recording, sleep briefly before checking again
            else:
                time.sleep(TIME_SLEEP)

    def _is_silero_speech(self, chunk):
        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(
                pcm_data, 16000, self.sample_rate)
            chunk = data_16000.astype(np.int16).tobytes()

        self.silero_working = True
        audio_chunk = np.frombuffer(chunk, dtype=np.int16)
        audio_chunk = audio_chunk.astype(np.float32) / INT16_MAX_ABS_VALUE
        vad_prob = self.silero_vad_model(
            torch.from_numpy(audio_chunk),
            SAMPLE_RATE).item()
        is_silero_speech_active = vad_prob > (1 - self.silero_sensitivity)
        if is_silero_speech_active:
            self.is_silero_speech_active = True
        self.silero_working = False
        return is_silero_speech_active

    def _is_webrtc_speech(self, chunk, all_frames_must_be_true=False):
        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(
                pcm_data, 16000, self.sample_rate)
            chunk = data_16000.astype(np.int16).tobytes()

        # Number of audio frames per millisecond
        frame_length = int(16000 * 0.01)  # for 10ms frame
        num_frames = int(len(chunk) / (2 * frame_length))
        speech_frames = 0

        for i in range(num_frames):
            start_byte = i * frame_length * 2
            end_byte = start_byte + frame_length * 2
            frame = chunk[start_byte:end_byte]
            if self.webrtc_vad_model.is_speech(frame, 16000):
                speech_frames += 1
                if not all_frames_must_be_true:
                    return True
        if all_frames_must_be_true:
            return speech_frames == num_frames
        else:
            return False

    def _check_voice_activity(self, data):
        self.is_webrtc_speech_active = self._is_webrtc_speech(data)

        # First quick performing check for voice activity using WebRTC
        if self.is_webrtc_speech_active:

            if not self.silero_working:
                self.silero_working = True

                # Run the intensive check in a separate thread
                threading.Thread(
                    target=self._is_silero_speech,
                    args=(data,)).start()

    def _is_voice_active(self):
        return self.is_webrtc_speech_active and self.is_silero_speech_active

    def _set_state(self, new_state):
        # Check if the state has actually changed
        if new_state == self.state:
            return

        # Store the current state for later comparison
        old_state = self.state

        # Update to the new state
        self.state = new_state

        # Execute callbacks based on transitioning FROM a particular state
        if old_state == "listening":
            if self.on_vad_detect_stop:
                self.on_vad_detect_stop()

        # Execute callbacks based on transitioning TO a particular state
        if new_state == "listening":
            self._set_spinner("speak now")
            if self.spinner and self.halo:
                self.halo._interval = 250
        elif new_state == "transcribing":
            self._set_spinner("transcribing")
            if self.spinner and self.halo:
                self.halo._interval = 50
        elif new_state == "recording":
            self._set_spinner("recording")
            if self.spinner and self.halo:
                self.halo._interval = 100
        elif new_state == "inactive":
            if self.spinner and self.halo:
                self.halo.stop()
                self.halo = None

    def _set_spinner(self, text):
        if self.spinner:
            # If the Halo spinner doesn't exist, create and start it
            if self.halo is None:
                self.halo = halo.Halo(text=text)
                self.halo.start()
            # If the Halo spinner already exists, just update the text
            else:
                self.halo.text = text

    def _preprocess_output(self, text, preview=False):
        text = re.sub(r'\s+', ' ', text.strip())

        if text:
            text = text[0].upper() + text[1:]

        # Ensure the text ends with a proper punctuation
        # if it ends with an alphanumeric character
        if not preview:
            if text and text[-1].isalnum():
                text += '.'

        return text

    def _find_tail_match_in_text(self, text1, text2, length_of_match=10):

        # Check if either of the texts is too short
        if len(text1) < length_of_match or len(text2) < length_of_match:
            return -1

        # The end portion of the first text that we want to compare
        target_substring = text1[-length_of_match:]

        # Loop through text2 from right to left
        for i in range(len(text2) - length_of_match + 1):
            # Extract the substring from text2
            # to compare with the target_substring
            current_substring = text2[len(text2) - i - length_of_match:
                                      len(text2) - i]

            # Compare the current_substring with the target_substring
            if current_substring == target_substring:
                # Position in text2 where the match starts
                return len(text2) - i

        return -1

    def _on_realtime_transcription_update(self, text):
        if self.on_realtime_transcription_update:
            if self.is_recording:
                self.on_realtime_transcription_update(text)

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()



if __name__=='__main__' :

    def process_text(text):
        print(text)

    recorder_config = {
        'spinner': False,
        'model': 'large-v2',
        'language': 'en',
        'silero_sensitivity': 0.4,
        'webrtc_sensitivity': 2,
        'post_speech_silence_duration': 0.4,
        'min_length_of_recording': 0,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
    }
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

    recorder = AudioWhisper(**recorder_config, input_device_index=audio_input_index)
    print("Say something...", end="", flush=True)

    while True:
        recorder.text(process_text)