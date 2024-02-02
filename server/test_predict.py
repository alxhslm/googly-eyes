import requests
from PIL import Image

url = "http://127.0.0.1:8502/identify_faces"
files = {"image": open("photo.jpg", "rb")}
response = requests.post(url, files=files)
response.raise_for_status()

url = "http://127.0.0.1:8502/googly_eyes"
settings = {"eye_size": 0.5, "pupil_size_range": [0.4, 0.6]}
response = requests.post(url, json=settings, files=files)
image = Image.open(response.content)
Image.save("photo_with_eyes.jpg")
