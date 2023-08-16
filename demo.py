import os
import torch
import torchvision.transforms as transforms
from model import resnet50
from utils import MyDataSet
from torch.utils.data import DataLoader
from attack import StatAttack
import torchvision.transforms.functional as TF
import argparse

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    data_loader = DataLoader(MyDataSet(args.fake_path, args.fake_label, trans),
                             batch_size=args.batch_size, shuffle=True)
    guidedloader = DataLoader(MyDataSet(args.real_path, args.real_label, trans),
                              batch_size=args.batch_size, shuffle=True)

    model = resnet50().to(device).eval()
    model.hook_middle_representation()

    atk = StatAttack(model, step=args.attack_step)
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for (data, labels), (guideImages, guideLabel) in zip(data_loader, guidedloader):
        data, labels = data.to(device), labels.to(device)
        adv_images = atk(guideImages, data)
        for idx, img in enumerate(adv_images):
            TF.to_pil_image(img).save(os.path.join(args.result_dir, f'result_{idx}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Attack Demo')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--real_path', default='./real', help='Path to real images')
    parser.add_argument('--real_label', type=int, default=0, help='Label for real images')
    parser.add_argument('--fake_path', default='./fake', help='Path to fake images')
    parser.add_argument('--fake_label', type=int, default=1, help='Label for fake images')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for data loading')
    parser.add_argument('--attack_step', type=int, default=10, help='Step for BiasFieldAttack')
    parser.add_argument('--result_dir', default='./result', help='Directory to save results')
    args = parser.parse_args()
    main(args)
