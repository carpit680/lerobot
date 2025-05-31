# kokoro_tts.py

from kokoro import KPipeline
from pydub import AudioSegment
import numpy as np
import sys
import subprocess
import tempfile
from io import BytesIO

class KokoroTTS:
    """
    A TTS wrapper class for the Kokoro pipeline.
    It converts model outputs (tensor or numpy array of floats in [-1, 1])
    to PCM audio and plays the audio using a platform-appropriate method.
    """
    def __init__(self, lang_code='a', sample_rate=22050):
        """
        Initialize the Kokoro TTS engine.

        :param lang_code: Language code (e.g., 'a' for American English).
        :param sample_rate: Sample rate of the output audio.
        """
        self.pipeline = KPipeline(lang_code=lang_code)
        self.sample_rate = sample_rate

    def _convert_tensor_to_audio_segment(self, audio):
        """
        Convert the audio (tensor or numpy array of floats in [-1, 1]) 
        to a pydub AudioSegment using 16-bit PCM.

        :param audio: Tensor or numpy array of floats.
        :return: A pydub AudioSegment object.
        """
        # Convert tensor to a numpy array if needed.
        if hasattr(audio, "numpy"):
            audio = audio.numpy()

        # Ensure audio is within the valid range.
        audio = np.clip(audio, -1, 1)

        # Scale to 16-bit PCM.
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Create an AudioSegment from the raw PCM data.
        segment = AudioSegment.from_raw(
            BytesIO(audio_bytes),
            sample_width=2,            # 2 bytes for 16-bit audio
            frame_rate=self.sample_rate,
            channels=1                 # Mono audio
        )
        return segment

    def _play_audio_segment(self, segment):
        """
        Play a pydub AudioSegment using a platform-specific method.
        
        On macOS, the audio is exported as a temporary WAV file and played using 'afplay'.
        On other platforms, it uses pydub's playback (which typically uses simpleaudio).

        :param segment: The pydub AudioSegment to play.
        """
        if sys.platform == "darwin":
            # macOS: export to a temporary WAV file and play using afplay.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_filename = tmp_file.name
                segment.export(temp_filename, format="wav")
            subprocess.run(["afplay", temp_filename])
        else:
            # Other systems: use pydub.playback.play
            from pydub.playback import play
            play(segment)

    def speak(self, text, voice='af_heart', speed=1, split_pattern=r'\n+'):
        """
        Generate and play audio for the given text using the Kokoro pipeline.
        
        :param text: The text to synthesize.
        :param voice: The voice identifier to use (e.g., 'af_heart').
        :param speed: The speech speed.
        :param split_pattern: A regex pattern to split the text into chunks.
        """
        # Generate audio data (yields tuples of (graphemes, phonemes, audio)).
        generator = self.pipeline(
            text,
            voice=voice,
            speed=speed,
            split_pattern=split_pattern
        )

        for i, (gs, ps, audio) in enumerate(generator):
            print(f"Chunk {i}:")
            print("Text:", gs)
            print("Phonemes:", ps)
            try:
                segment = self._convert_tensor_to_audio_segment(audio)
                self._play_audio_segment(segment)
            except Exception as e:
                print("Error playing audio:", e)


if __name__ == "__main__":
    # Example usage: Run this module directly to test TTS playback.
    sample_text = '''
    The sky above the port was the color of television, tuned to a dead channel.
    "It's not like I'm using," Case heard someone say, as he shouldered his way through the crowd around the door of the Chat.
    "It's like my body's developed this massive drug deficiency."
    '''
    tts = KokoroTTS(lang_code='a', sample_rate=22050)
    tts.speak(sample_text)
