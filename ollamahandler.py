import requests
import json
import time

# URL of the API endpoint
url = 'http://127.0.0.1:11434/api/chat'

headers = {
    "User-Agent": "ollama/0.1.48 (arm64 darwin) Go/go1.22.0",
    "Accept": "application/x-ndjson",
    "Content-Type": "application/json",
}

character_personality = """
You are a passionate software developer and you have vast knowledge of 
all technologies java, aws, kafka, kubernetes, docker, sql, multithreading, rest apis.
You are attending an interview for software developer role. 
"""

# Create a session object to maintain context
# session = requests.Session()

def loadModel() :
    # Initial request payload to load the model
    # global session
    initial_payload = {
        "model": "llama3:8b",
        "messages": None,
        "format": "",
        "options": None
    }

    # Sending the initial request
    initial_response = requests.post(url, headers=headers, data=json.dumps(initial_payload))
    if initial_response.status_code == 200:
        print("Model loaded successfully.")
    else:
        print("Error loading model:", initial_response.status_code, initial_response.text)
        exit()

# Function to send the request and process responses
def process(text):
    global url, headers
    payload = {
        "model": "llama3:8b",
        "messages": [
            # {"role": "system", "content": character_personality},
            {"role": "user", "content": f"Hey explain, {text}"}
            ],
        "format": "",
        "options": {}
    }
    return requests.post(url, headers=headers, json=payload, stream=True)

def testOllamaModel():
    start_time = time.time()
    loadModel()
    load_time = time.time() - start_time
    print(f"Time taken to load model: {load_time:.2f} seconds")
    for i in range(3):
        start_time = time.time()
        response = process(f"text white {i}")
        process_time = time.time() - start_time
        output = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                output += data['message']['content']
        print(output)
        print(f"Time taken for process {i+1} : {process_time:.2f} seconds")

def ollamaStop() :
    global url, headers
    payload = {
        "model": "llama3:8b",
        "messages": [
            {"role": "user", "content": "/bye"}
            ],
        "format": "",
        "options": {}
    }
    # session.post(url, headers=headers, json=payload)
    # session.close()