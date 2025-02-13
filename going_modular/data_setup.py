
import os
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
NUM_WORKER = os.cpu_count()
def create_dataloader(train_dir : str,
                      test_dir : str,
                      transform : transforms.Compose,
                      batch_size : int,
                      num_workers : int = NUM_WORKER):
  train_data = datasets.ImageFolder(train_dir, transform = transform)
  test_data = datasets.ImageFolder(test_dir, transform = transform)
  class_name = train_data.classes
  train_dataloader = DataLoader(train_data,
                                batch_size = batch_size,
                                shuffle = True,
                                pin_memory = True,
                                num_workers = num_workers)
  test_dataloader = DataLoader(test_data,
                               batch_size = batch_size,
                               shuffle = False,
                               pin_memory = True,
                               num_workers = num_workers)
  return train_dataloader, test_dataloader, class_name 
