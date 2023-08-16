import torch
from torch import nn
import timm
import torchvision.transforms as transforms
import os
from utils import MyDataSet
from torch.utils.data import DataLoader


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


class resnet50(nn.Module):
    def __init__(self, use_cuda=True):
        super(resnet50, self).__init__()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        #norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        res = timm.create_model('resnet50', num_classes=1)
        resPath = '/houyang/ns235x/weights/blur_jpg_prob0.5.pth'
        res_state_dict = torch.load(resPath)
        res.load_state_dict(res_state_dict['model'])

        self.model = nn.Sequential(
            #norm_layer,
            res
        ).to(self.device)
        self.features = {}

    def hook_middle_representation(self):
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        self.model[0].layer4.register_forward_hook(get_features('conv_plr'))
        self.model[0].global_pool.register_forward_hook(get_features('global_pool'))


    def forward(self, x):
        return self.model(x.to(self.device))



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    real_path = '/houyang/DataSet/newData/testData/stargan/0_real'
    real_label = 0
    fake_path = '/houyang/DataSet/newData/testData/stargan/1_fake'
    fake_label = 1

    trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
    ])

    loadData = MyDataSet(real_path, real_label, trans)
    data_loader = DataLoader(loadData, batch_size=20, shuffle=True, pin_memory=False, num_workers=0)

    images, labels = iter(data_loader).next()
    print(labels)
    print(images.shape)
    model = resnet50()
    model.eval()
    model.hook_middle_representation()
    output = model(images)
    print(output)
    print(output.shape)
    print(model.features.keys())
    print(model.features['conv_plr'].flatten(1).shape)




