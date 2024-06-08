import requests

def generate_text(input_text):
    response = requests.post("http://127.0.0.1:8000/generate/", json={"text": input_text})
    if response.status_code == 200:
        return response.json()["generated_text"]
    else:
        return "Error: Unable to generate text"

if __name__ == "__main__":
    input_text = input("Enter the text to start with: ")
    generated_text = generate_text(input_text)
    print("Generated text:", generated_text)
