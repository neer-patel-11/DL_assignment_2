"""Training entrypoint
"""
from data.pets_dataset import get_data_loader
from models.classification import VGG11Classifier
import torch.optim as optim 
import torch.nn as nn
def train_vgg11(batch_size = 32 , lr=0.5):
    train_loader , test_loader = get_data_loader()

    model = VGG11Classifier()
    
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    criterion = nn.CrossEntropyLoss()
    

    for i, (x,y) in enumerate(train_loader):

        outputs = model(x)

        loss = criterion(outputs,y)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()          
        
        print(f"bacth {i+1}, Loss: {loss.item():.4f}")






if __name__ == "__main__":

    train_vgg11()

