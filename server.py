import litserve as ls
from fastapi import Response
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import time, os

class MusicGenLitAPI(ls.LitAPI):

    def setup(self, device):
        self.model = MusicGen.get_pretrained('facebook/musicgen-stereo-melody-large')
        self.model.set_generation_params(duration=25)

    def decode_request(self, request):
        path = f"tmp/input_{time.time()}"
        with open(path, "wb") as f:
            f.write(request["content"].file.read())

        prompt = request["prompt"]
        return {
            "path": path, 
            "prompt": prompt
            }

    def predict(self, params):
        melody, sr = torchaudio.load(params["path"])
        os.remove(params["path"])

        wav = self.model.generate_with_chroma([params["prompt"]], melody, sr)

        for idx, one_wav in enumerate(wav):
            path = audio_write(f'tmp/output_{time.time()}', one_wav.cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True)
            with open(path, "rb") as f:
                data = f.read()
            os.remove(path)
            return data

    def encode_response(self, prediction):
        return Response(content=prediction, headers={"Content-Type": "audio/wav"})

if __name__ == "__main__":
    api = MusicGenLitAPI()
    server = ls.LitServer(api, accelerator="gpu", timeout=10000, workers_per_device=2)
    server.run(port=8000)