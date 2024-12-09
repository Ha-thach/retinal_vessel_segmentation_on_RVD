import torch
import torch.nn as nn


class RRCNN_block(nn.Module):
    def __init__(self, in_c, out_c, t=2, dropout_rate=0.5):
        super().__init__()
        self.t = t
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=self.dropout_rate)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.recurrent_block = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout here

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout here

        for _ in range(self.t):
            x = self.recurrent_block(x + x)

        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, t=2, dropout_rate=0.5):
        super().__init__()
        self.conv = RRCNN_block(in_c, out_c, t, dropout_rate)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, F_int, dropout_rate=0.5):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.att = Attention_block(F_g=out_c, F_l=out_c, F_int=F_int)
        self.conv = RRCNN_block(out_c + out_c, out_c, dropout_rate=dropout_rate)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = self.att(g=x, x=skip)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class R2AttUnet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64, t=2, dropout_rate=dropout_rate)
        self.e2 = encoder_block(64, 128, t=2, dropout_rate=dropout_rate)
        self.e3 = encoder_block(128, 256, t=2, dropout_rate=dropout_rate)
        self.e4 = encoder_block(256, 512, t=2, dropout_rate=dropout_rate)

        """ Bottleneck """
        self.b = RRCNN_block(512, 1024, t=2, dropout_rate=dropout_rate)

        """ Decoder """
        self.d1 = decoder_block(1024, 512, F_int=256, dropout_rate=dropout_rate)
        self.d2 = decoder_block(512, 256, F_int=128, dropout_rate=dropout_rate)
        self.d3 = decoder_block(256, 128, F_int=64, dropout_rate=dropout_rate)
        self.d4 = decoder_block(128, 64, F_int=32, dropout_rate=dropout_rate)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs


if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))
    f = R2AttUnet(dropout_rate=0.5)  # Pass dropout rate to the model
    y = f(x)
    print(y.shape)

