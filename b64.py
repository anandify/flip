import base64

# Replace 'your_image.jpg' with the path to your image file
image_path = "orange.jpg"

# Encode the image to Base64 format
with open(image_path, "rb") as image_file:
    base64_encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

print(base64_encoded_image)  # Output the Base64-encoded string
