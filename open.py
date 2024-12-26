import requests
import json

# Your NVIDIA API credentials
api_key = "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"
url = "https://integrate.api.nvidia.com/v1/chat/completions"

# Headers for authentication
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Define your data payload for the model
data = {
    "model": "meta/llama-3.1-405b-instruct",
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "temperature": 0.2,
    "top_p": 0.7,
    "max_tokens": 1024,
    "stream": True
}

# Send the POST request to the NVIDIA API
response = requests.post(url, headers=headers, data=json.dumps(data))

# Check if the response was successful
if response.status_code == 200:
    # Process the response in chunks (since we're streaming)
    for chunk in response.iter_lines():
        if chunk:
            try:
                json_chunk = json.loads(chunk)
                if 'choices' in json_chunk and json_chunk['choices'][0].get('delta', {}).get('content'):
                    print(json_chunk['choices'][0]['delta']['content'], end="")
            except json.JSONDecodeError:
                pass
else:
    print(f"Error: {response.status_code} - {response.text}")



