import requests

flight = {"Airline": "IndiGo", "Source":"Banglore", "Destination":"New Delhi", "Total_Stops":2}		


url = 'http://localhost:9696/predict'
response = requests.post(url, json=flight)
print(response.json())
