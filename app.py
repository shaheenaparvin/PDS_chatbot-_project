import sys
sys.path.append('/Users/parvin/Desktop/torch_bot')
from fastapi import Body, FastAPI, Request
import json
import torch
import torch.nn as nn
from ml.model import NeuralNet
from utils.utils import tokenize, bag_of_words
import random
from models import Message
from uuid import uuid4

## loading ml model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('../shared/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "../shared/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

app = FastAPI(title="Doc Assistant APIs")

@app.get('/')
def health(request: Request):
    return {"message": "Hello from Doc Assistant APIs", "Version": "1.0"}


@app.post('/send')
def send_message(request: Request, body: Message):
    sentence = tokenize(body.message)
    X = bag_of_words(body.message, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    if tag == "booking":
        return {"type": "Booking", "via": "Bot", "message": ""}

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(tag, prob.item())
    if prob.item() > 0.3:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                d = {"message": f"{random.choice(intent['responses'])}", "type": "Direct", "via": "Bot", "id": uuid4()}
                print(d)
                return d
    else:
        return {"message": "I do not understand...", "type": "Direct", "via": "Bot", 'id': uuid4()}


@app.post('/book-appointment')
def book_appointment(request: Request):
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app)
