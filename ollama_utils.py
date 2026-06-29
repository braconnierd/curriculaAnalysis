import subprocess
import time

#Additional functions to manage Ollama on MacOS

def ensure_ollama_running(wait_time: int = 2):
    """
    Ensure the Ollama server is running.
    If not, start it in the background.

    Args:
        wait_time (int): How many seconds to wait after starting Ollama.
    """
    OLLAMA_PATH = "/opt/homebrew/bin/ollama"  # Adjust to your own path to Ollama if needed

    try:
        # Check if Ollama process is already running
        result = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Ollama not running — starting it now...")
            subprocess.Popen(
                [OLLAMA_PATH, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(wait_time)
            print("Ollama started successfully.")
        else:
            print("Ollama is already running.")
    except FileNotFoundError:
        print(f"Ollama not found at {OLLAMA_PATH}. Check your installation path.")
    except Exception as e:
        print(f"Error while checking/starting Ollama: {e}")


def is_ollama_running() -> bool:
    """
    Check whether Ollama is currently running.
    Returns:
        bool: True if running, False otherwise.
    """
    try:
        result = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True)
        return result.returncode == 0
    except Exception:
        return False
    

def stop_ollama():
    """
    Stop the Ollama server if it's running in the background.
    """
    try:
        if is_ollama_running():
            subprocess.run(["pkill", "-x", "ollama"])
            print("Ollama stopped successfully.")
        else:
            print("Ollama is not currently running.")
    except Exception as e:
        print(f"Error while stopping Ollama: {e}")

ensure_ollama_running()