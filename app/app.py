import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models

app = FastAPI()

# set structure
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        
        self.name = "ANN"
        self.fc1 = nn.Linear(256 * 6 * 6, 200)
        self.fc2 = nn.Linear(200, 3)

    def forward(self, x):
        x = x.view(-1, 256 * 6 * 6) #flatten feature data
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

alexnet = torchvision.models.alexnet(pretrained=True)

def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,batch_size,learning_rate,epoch)
    return path

net = ANN()
model_path = get_model_name(net.name, batch_size=32, learning_rate=0.002, epoch=4)
state = torch.load("model_ANN_bs32_lr0.002_epoch4",map_location=torch.device('cpu'))
net.load_state_dict(state)

# Load the trained model
model = net

# Define the API endpoint
app = FastAPI()

"""@app.post("/")
def read_root():
    return {"hello":"ff"}"""

@app.post("/")
async def predict(file: UploadFile = File(...)):
     # Read the image file in memory
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize([224,224]),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = transform(image)

    imgs = alexnet.features(image) 
    output = net(imgs)
    pred = output.max(1, keepdim=False)[1]
    pred = pred.tolist()
    if pred[0] == 0:
        predicted_class= "Bed"
    elif pred[0] == 1:
        predicted_class =  "Chair"
    else:
        predicted_class =  "Sofa"

    # Return the predicted class label as a JSON response
    return JSONResponse({"predicted_class": predicted_class})



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


"""@app.post("http://0.0.0.0:8000")
async def predict(file: UploadFile = File(...)):

    # Read the image file in memory
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize([224,224]),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = transform(image)

    imgs = alexnet.features(image) 
    output = net(imgs)
    pred = output.max(1, keepdim=False)[1]
    pred = pred.tolist()
    if pred[0] == 0:
        predicted_class= "Bed"
    elif pred[0] == 1:
        predicted_class =  "Chair"
    else:
        predicted_class =  "Sofa"

    # Return the predicted class label as a JSON response
    return JSONResponse({"predicted_class": predicted_class})"""


