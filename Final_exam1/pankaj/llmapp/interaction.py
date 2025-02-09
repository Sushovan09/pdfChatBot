import requests

# Assuming your Flask server is running locally on port 5000
base_url = 'http://localhost:5000'

# Initialize the chatbot by sending a POST request to /initialize endpoint
initialize_url = f'{base_url}/initialize'
response = requests.post(initialize_url)
print(response.json())  # Print the response from the server

# Ask a question by sending a POST request to /ask endpoint
ask_url = f'{base_url}/ask'
question = "what is Block Chain Technology"
data = {'question': question}
response = requests.post(ask_url, json=data)
print(response.json())  # Print the response from the server

