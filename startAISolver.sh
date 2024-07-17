#!/bin/sh

# Step 1: Check if Ollama is installed, if not, install it
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
    ollama --version
fi

# Step 2: Pull the llama3:8b model
echo "Pulling llama3:8b model..."
ollama pull llama3:8b

# Step 4: Install Python requirements
echo "Installing Python requirements..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "Python requirements installed successfully."
else
    echo "Failed to install Python requirements."
    exit 1
fi

# Function to clear the screen based on the operating system
clear_screen() {
    case "$(uname)" in
        Darwin)
            clear  # macOS
            ;;
        Linux)
            clear  # Linux
            ;;
        CYGWIN*|MINGW32*|MSYS*|MINGW*)
            cls  # Windows
            ;;
        *)
            echo "Unsupported OS"
            ;;
    esac
}

# Call the function
clear_screen


# Step 5: Run the Python script
echo "Running main py..."
python SimpleGoogleSpeech.py --debug False
if [ $? -eq 0 ]; then
    echo "Python script ran successfully."
else
    echo "Failed to run Python script."
    exit 1
fi
