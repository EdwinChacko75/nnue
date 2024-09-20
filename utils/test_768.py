import h5py

file_path = '/home/edwin/nnue/768_processed.h5'
file_path = '/home/edwin/nnue/chess_data_processed.h5'

with h5py.File(file_path, 'r') as f:
    len = f['X_b'].shape
print(len)