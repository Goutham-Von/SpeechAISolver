import pyaudio
from loggingRtSTT import log
import torch.multiprocessing as mp
from RealtimeSTT import AudioToTextRecorder

class MockAudioStream :

    def __init__(self, audio_input_index, audioRecorder) :
        self.audio_input_index = audio_input_index
        self.recorder = audioRecorder
        self.audio_interface = pyaudio.PyAudio()
        self.passStream = True
        

    def read(self, buffSize) :
        if(not self.passStream) :
            self.stream.read(buffSize)

    def stop_stream(self, mock = True) :
        if(not mock) :
            self.stream.stop_stream()

    def close(self, mock = True):
        if(not mock) :
            self.stream.close()

    def mockAudioStream(self) :
        passStream = True

    def resetAudioStream(self) :
        global stream
        self.passStream = False
        stream = self.audio_interface.open(
            rate=self.recorder.sample_rate,
            format=pyaudio.paInt16,
            channels=1,
            input=True,
            frames_per_buffer=self.recorder.buffer_size,
            input_device_index=self.recorder.input_device_index
        )

    def test() :
        log('testing mock audio stream')
        print('testing mock audio stream')

def resetAudioStream(recorder) :
    stopAudioStream(recorder)
    recorder.reader_process = mp.Process(
                target=AudioToTextRecorder._audio_data_worker,
                args=(
                    recorder.audio_queue,
                    recorder.sample_rate,
                    recorder.buffer_size,
                    recorder.input_device_index,
                    recorder.shutdown_event,
                    recorder.interrupt_stop_event,
                    recorder.use_microphone
                )
            )
    # recorder.shutdown_event.clear()
    recorder.reader_process.start()

def stopAudioStream(recorder) :
    # recorder.shutdown_event.set()
    if(recorder.reader_process.is_alive()) :
        recorder.reader_process.kill()