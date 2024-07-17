#!/bin/bash

# Run the ngrok.py script to get the token
TOKEN=$(python3 -c 'import NgrokEnvironmentSetter as ngrok; print(ngrok.get_token())')

# Check if the TOKEN variable is not empty
if [ -z "$TOKEN" ]; then
  echo "Failed to retrieve token from ngrok.py"
  exit 1
fi

# Export the token as an environment variable
export OLLAMA_HOST=$TOKEN
echo "OLLAMA_HOST set to $OLLAMA_HOST"

# Run main program
python3 speech-vosk.py
