import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from nnue.data_loader import CustomDataset
from nnue.nnue import NNUE
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '0,1,2,4'

FILE_PATH = '/home/edwin/nnue/data/768_processed.h5'
NUM_FEATURES = 64*6*2
FEATURE_PROJ_SIZE = 4
CAT_LAYER_SIZE = 8
FINAL_LAYER_SIZE = 8
BATCH_SIZE = 8192*2
EPOCHS = 10
LEARNING_RATE = 0.01

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # os.environ['CUDA_VISIBLE_DEVICES'] =  '0,1,2,4'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    dataset = CustomDataset(FILE_PATH, NUM_FEATURES)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, num_workers=4, sampler=train_sampler, batch_size=BATCH_SIZE)

    # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    # val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=BATCH_SIZE)


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

    if rank == 0:
        torch.save(model.module.state_dict(), "nnue_model.pth")
        print("Training completed. Model saved.")

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()