import argparse
import requests
from datetime import datetime

# Update this URL to your server's URL if hosted remotely
API_URL = "http://127.0.0.1:8000/predict"

def send_generate_request(prompt, path):
    inputFile = open(path, 'rb')
    inputData = inputFile.read()
    inputFile.close()

    response = requests.post(API_URL, files={"prompt": (None, prompt), "content": inputData})
    if response.status_code == 200:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S").lower()
        filename = f"{prompt.replace(' ', '_')}.wav"
        
        with open(filename, "wb") as audio_file:
            audio_file.write(response.content)
        
        print(f"Audio saved to {filename}")
    else:
        print(f"Error: Response with status code {response.status_code} - {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send text to stable audio server and receive generated audio.")
    parser.add_argument("--prompt", required=True, help="Prompt to generate audio.")
    parser.add_argument("--path", required=True, help="Path for the file's melody")
    args = parser.parse_args()
    
    send_generate_request(args.prompt, args.path)