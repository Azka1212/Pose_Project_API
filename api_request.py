import requests

url = "http://127.0.0.1:8000/estimate-pose/"

files = {
    'front_image': open('files/8/front_img.jpg', 'rb'),
    'side_image': open('files/8/side_img.jpg', 'rb')
}

data = {
    'height': '167'
}

response = requests.post(url, files=files, data=data)

print(response.json())
