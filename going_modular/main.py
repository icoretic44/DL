from going_modular import engine, model_builder, data_setup, utils
from torch import nn
import torch
import torchvision
from torchvision import transforms
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"
transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor()
                                ])
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
model1 = model_builder.TinyVGG(3,10,3).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr = 0.001)
batch_size = 32
train(model1,train_dir,test_dir,batch_size,epochs=10,transform = transform,loss_fn = loss_fn,optimizer = optimizer,device = device)
