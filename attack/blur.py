import torch.nn as nn
import torch

class AdaptiveGaussianFilter(nn.Module):
    def __init__(self, filter_size, img_dim=256):
        super().__init__()
        assert (filter_size % 2) > 0, f"Filter size must be odd. Received: {filter_size}"

        self.filter_size = filter_size
        self.image_size = img_dim
        self.reflection_padding = nn.ReflectionPad2d(int((self.filter_size - 1) // 2))
        self.base_weights = [[None] * self.filter_size for _ in range(self.filter_size)]
        mid_point = filter_size // 2

        for row in range(self.filter_size):
            for col in range(self.filter_size):
                self.base_weights[row][col] = torch.full((1, 3, img_dim, img_dim),
                                                         -((row - mid_point) * (row - mid_point) + (col - mid_point) * (col - mid_point)) / 2,
                                                         dtype=torch.float32).cuda()

    def forward(self, x, sigma):
        self.tmp_weight = self.weights_init(sigma, x)
        result = torch.zeros_like(x)
        x = self.reflection_padding(x)
        for row in range(self.filter_size):
            for col in range(self.filter_size):
                result += x[:, :, row:row + self.image_size, col:col + self.image_size] * self.tmp_weight[row][col]
        return result

    def weights_init(self, sigma, x):
        weights = [[None] * self.filter_size for _ in range(self.filter_size)]
        for row in range(self.filter_size):
            for col in range(self.filter_size):
                weights[row][col] = torch.exp(self.base_weights[row][col].expand(x.shape[0], -1, -1, -1) * sigma * sigma).to(sigma.device)
                if row == 0 and col == 0:
                    weights_sum = weights[row][col].clone()
                else:
                    weights_sum += weights[row][col]

        for row in range(self.filter_size):
            for col in range(self.filter_size):
                weights[row][col] = weights[row][col] / weights_sum
        return weights






