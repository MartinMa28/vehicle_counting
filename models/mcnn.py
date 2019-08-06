import torch
import torch.nn as nn

class MCNN(nn.Module):
    """
    The model of multi-column CNN - MCNN
    """
    def __init__(self, load_weights=False):
        super(MCNN, self).__init__()

        self.branch_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )

        self.branch_2 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(20, 40, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(40, 20, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 10, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch_3 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Conv2d(30, 1, kernel_size=1, padding=0)

        if not load_weights:
            self.apply(self._initialize_weights)

    
    def forward(self, img):
        x1 = self.branch_1(img)
        x2 = self.branch_2(img)
        x3 = self.branch_3(img)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fuse(x)
        
        return x

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)


# test
if __name__ == '__main__':
    dummy_img = torch.rand((8, 3, 240, 352), dtype=torch.float)
    mcnn = MCNN()
    dummy_dm = mcnn(dummy_img)
    print(dummy_dm.shape)