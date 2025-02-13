from going_modular import engine, model_builder, data_setup, utils
from torch import nn
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"
transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor()
                                ])
torch.manual_seed(42)
model1 = model_builder.TinyVGG(3,10,3).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr = 0.001)
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
train_dataloader,test_dataloader,class_name = data_setup.create_dataloader(train_dir,
                                                                           test_dir,
                                                                           transform = transform,
                                                                           batch_size = batch_size)
results = engine.train(model = model1,
                train_dataloader= train_dataloader,
                test_dataloader= test_dataloader,
                epochs = 5,
                loss_fn = loss_fn,
                optimizer = optimizer,
                device = device)
utils.save_model(model=model1,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
