from going_modular import engine, model_builder, data_setup, utils
from torch import nn
import torch
import torchvision
from torchvision import transforms
def train(model:torch.nn.Module,
          train_dir : str,
          test_dir : str,
          batch_size : int,
          epochs : int,
          transform : transforms.Compose,
          loss_fn : torch.nn.Module,
          optimizer : torch.optim.Optimizer,
          device : torch.device):
  torch.manual_seed(42)
  train_dataloader,test_dataloader,class_name = data_setup.create_dataloader(train_dir,
                                                                           test_dir,
                                                                           transform = transform,
                                                                           batch_size = batch_size)
  results = engine.train(model = model,
                train_dataloader= train_dataloader,
                test_dataloader= test_dataloader,
                epochs = epochs,
                loss_fn = loss_fn,
                optimizer = optimizer,
                device = device)
  utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
  
