import chess
import h5py
import numpy as np
from tqdm import tqdm

def halfkp_idx(king_square, square, piece_type, color, num_features):
    idx = square + 64 * piece_type + 64 * 6 * color
    if idx >= num_features:
        raise ValueError(f"Index out of range: {idx}")
    return idx

def fen_to_vector(board, num_features):
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    
    ones = []

    for square in chess.SQUARES:
        if square in (white_king_square, black_king_square):
            continue

        piece = board.piece_at(square)
        if piece:
            color = piece.color
            king_square = white_king_square if color == chess.WHITE else black_king_square
            idx = halfkp_idx(king_square, square, piece.piece_type - 1, color, num_features)
            ones.append(idx)
    
    return ones

def process_batch(dataset, start, end, num_features):
    X_w, X_b, y = [], [], []
    
    for i in range(start, end):
        try:
            fen_bytes, evaluation = dataset[i]
            fen_string = fen_bytes.decode('utf-8')
            
            board = chess.Board(fen_string)
            w_ones = fen_to_vector(board, num_features)
            
            mirrored_board = board.copy()
            mirrored_board.mirror()
            b_ones = fen_to_vector(mirrored_board, num_features)
            
            X_w.append(w_ones)
            X_b.append(b_ones)
            y.append(evaluation)
        except Exception as e:
            print(f"Error processing entry {i}: {e}")
    
    return X_w, X_b, y

def save_batch_to_hdf5(f, X_w, X_b, y, batch_start):
    X_w = np.array(X_w, dtype=object)
    X_b = np.array(X_b, dtype=object)
    y = np.array(y, dtype=np.float32)
    
    f['X_w'].resize((batch_start + X_w.shape[0],))
    f['X_b'].resize((batch_start + X_b.shape[0],))
    f['y'].resize((batch_start + y.shape[0],))

    f['X_w'][batch_start:batch_start + X_w.shape[0]] = X_w
    f['X_b'][batch_start:batch_start + X_b.shape[0]] = X_b
    f['y'][batch_start:batch_start + y.shape[0]] = y

def process_data_in_batches(input_file_path, output_file_path, batch_size, num_features):
    with h5py.File(input_file_path, 'r') as input_file:
        dataset = input_file['evaluations']
        total_samples = dataset.shape[0]
        
        with h5py.File(output_file_path, 'w') as output_file:
            # Create resizable datasets
            output_file.create_dataset('X_w', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.dtype('int32')))
            output_file.create_dataset('X_b', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.dtype('int32')))
            output_file.create_dataset('y', shape=(0,), maxshape=(None,), dtype=np.float32)
            
            pbar = tqdm(total=total_samples, desc="Processing samples")
            for batch_start in range(0, total_samples, batch_size):
                batch_end = min(batch_start + batch_size, total_samples)
                
                X_w, X_b, y = process_batch(dataset, batch_start, batch_end, num_features)
                save_batch_to_hdf5(output_file, X_w, X_b, y, batch_start)

                pbar.update(batch_end - batch_start)
            
            pbar.close()

            # Validate the processed data
            print("Validating processed data...")
            assert output_file['X_w'].shape[0] == total_samples, "Mismatch in X_w samples"
            assert output_file['X_b'].shape[0] == total_samples, "Mismatch in X_b samples"
            assert output_file['y'].shape[0] == total_samples, "Mismatch in y samples"
            print("Data validation successful.")
def main():
    # Parameters
    input_file_path = '/home/edwin/nnue/chess_data.h5'
    output_file_path = '/home/edwin/nnue/768_processed.h5'
    batch_size = 100000  # Adjust this based on your memory constraints
    num_features = 64 * 6 * 2

    # Run the batch processing and saving
    process_data_in_batches(input_file_path, output_file_path, batch_size, num_features)

    print("Processing complete. Verifying file integrity...")
    with h5py.File(output_file_path, 'r') as f:
        print(f"Total samples processed: {f['y'].shape[0]}")
        print(f"X_w shape: {f['X_w'].shape}")
        print(f"X_b shape: {f['X_b'].shape}")
        print(f"y shape: {f['y'].shape}")

    print("Script execution completed successfully.")

if __name__ == "__main__":
    main()