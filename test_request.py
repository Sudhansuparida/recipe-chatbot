import requests

query = {"query": "egg, onion"}
response = requests.post("http://127.0.0.1:8000/search", json=query)

print(response.json())


