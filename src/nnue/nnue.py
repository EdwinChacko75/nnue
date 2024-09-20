import torch

class NNUE(torch.nn.Module):
    def __init__(self, NUM_FEATURES, M, N, K):
        super(NNUE, self).__init__()

        self.feature_transformer = torch.nn.Linear(NUM_FEATURES, M)
        self.layer_1 = torch.nn.Linear(2 * M, N)
        self.layer_2 = torch.nn.Linear(N, K)
        self.layer_3 = torch.nn.Linear(K, 1)

    def forward(self, white_features, black_features, side_to_move=None):
        white = self.feature_transformer(white_features)
        black = self.feature_transformer(black_features)

        accumulator = torch.cat([white, black], dim = 1)
        
        # torch.clampe effectively achieves clipped ReLU
        l1_x = torch.clamp(accumulator, 0.0, 1.0)
        l2_logits = self.layer_1(l1_x)
        l2_x = torch.clamp(l2_logits, 0.0, 1.0)
        l3_logits = self.layer_2(l2_x)
        l3_3 = torch.clamp(l3_logits, 0.0, 1.0)
        logits = self.layer_3(l3_3)
        # outputs = torch.clamp(logits, 0.0, 1.0)

        return logits.squeeze(-1)
