import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from nnue.data_loader import CustomDataset
from nnue.nnue import NNUE
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] =  '0,1,2,4'

FILE_PATH = '/home/edwin/nnue/data/768_processed.h5'
SAVE_DIR = "/home/edwin/nnue/models/initial_run"

NUM_FEATURES = 64*6*2
FEATURE_PROJ_SIZE = 4
CAT_LAYER_SIZE = 8
FINAL_LAYER_SIZE = 8
BATCH_SIZE = 8192*2
EPOCHS = 10
LEARNING_RATE = 0.01
TRAIN_TEST_SPLIT = (0.7, 0.2, 0.1)
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # os.environ['CUDA_VISIBLE_DEVICES'] =  '0,1,2,4'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
def get_temp_split(split):
    temp = 1- split[0]
    test = split[1]/temp
    val = split[2]/temp
    return [[split[0], temp], [test, val]]
def train(rank, world_size):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    dataset = CustomDataset(FILE_PATH, NUM_FEATURES)
    num_data = len(dataset)
    indices = list(range(num_data))
    split_train = int(num_data * 0.7)
    split_val = int(num_data * 0.3)

    # Shuffle the indices
    np.random.shuffle(indices)

    # Create datasets using only a portion of the indices
    train_indices = indices[:split_train]
    val_indices = indices[split_train:split_train + split_val]
    test_indices = indices[split_train + split_val:]
    # reinitializing, dont do that. also continue attempting new indices method.
    train_dataset = CustomDataset(dataset, train_indices)
    val_dataset = CustomDataset(dataset, val_indices)
    test_dataset = CustomDataset(dataset, test_indices)

    # train_size = int(0.7 * len(dataset))
    # val_size = len(dataset) - train_size

    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train, test_val = get_temp_split(TRAIN_TEST_SPLIT)
    # print(train)
    # train_dataset, temp_dataset = torch.utils.data.random_split(dataset, train)
    # test_dataset, val_dataset = torch.utils.data.random_split(temp_dataset, test_val)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, num_workers=4, sampler=train_sampler, batch_size=BATCH_SIZE, pin_memory=True)



    model = NNUE(NUM_FEATURES, FEATURE_PROJ_SIZE, CAT_LAYER_SIZE, FINAL_LAYER_SIZE).to(device)
    model = DDP(model, device_ids=[rank])

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        for white_features, black_features, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", disable=rank != 0):
            white_features, black_features, y = white_features.to(device), black_features.to(device), y.to(device)
            outputs = model(white_features, black_features)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if rank == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")
            torch.save(model.module.state_dict(), f"/home/edwin/nnue/models/initial_run/epoch_{epoch+1}.pth")

    # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    # val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=BATCH_SIZE)
    # test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    # test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

    if rank == 0:
        save_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth")
        torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
