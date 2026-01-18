import os
from openai import OpenAI

# Get API key from environment variable for security
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set")
    print("Please set it using: export OPENAI_API_KEY='your-api-key-here'")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Using a valid model name
        messages=[
            {"role": "user", "content": "Write a one-sentence bedtime story about a unicorn."}
        ],
        max_tokens=100
    )
    
    print(response.choices[0].message.content)
    
except Exception as e:
    print(f"Error: {e}")
