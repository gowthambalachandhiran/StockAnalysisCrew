import os
from dotenv import load_dotenv

# Specify the path to the .env file using os.path.join
env_folder = "./config"  # Folder containing the .env file
env_file = ".env"       # Name of the .env file
env_path = os.path.join(env_folder, env_file)

# Load the .env file
load_dotenv(dotenv_path=env_path)

if not os.path.exists(env_path):
    raise FileNotFoundError(f".env file not found at: {env_path}")


# Access the GROQ_API_KEY from the environment
groq_api_key = os.getenv("GROQ_API_KEY")

# Raise an error if the key is not loaded
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set. Please check your .env file.")

print(f"GROQ_API_KEY loaded successfully: {groq_api_key[:4]}... (truncated)")


