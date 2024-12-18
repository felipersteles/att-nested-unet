import torch
from torch import nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Residual block
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        
        # Downsample layer for the skip connection if in_channels != out_channels or stride != 1
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Residual branch
        residual = self.residual_block(x)
        
        # Shortcut connection (identity or downsampled)
        shortcut = x if self.downsample is None else self.downsample(x)
        
        # Add the residual and shortcut
        out = residual + shortcut
        return self.relu(out)

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=1, deep_supervision=False):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Down-sampling layers
        self.conv0_0 = ResNetBlock(input_channels, nb_filter[0])
        self.conv1_0 = ResNetBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ResNetBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ResNetBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ResNetBlock(nb_filter[3], nb_filter[4])

        # Nested U-Net layers
        self.conv0_1 = ResNetBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ResNetBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ResNetBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = ResNetBlock(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = ResNetBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = ResNetBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = ResNetBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = ResNetBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = ResNetBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = ResNetBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        # Output layers
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        # Encoder
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Decoder
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # Output
        output = None

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)

            output =  [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
        
        return output

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        add_xg = self.relu(theta_x + phi_g)
        psi = self.sigmoid(self.psi(add_xg))
        psi_upsampled = self.upsample(psi)
        return x * psi_upsampled

class NestedUNetWithAttention(NestedUNet):
    def __init__(self, num_classes, input_channels=1, deep_supervision=False):
        super().__init__(num_classes, input_channels, deep_supervision)

        nb_filter = [64, 128, 256, 512, 1024]
        self.attn1 = AttentionBlock(nb_filter[0], nb_filter[1], nb_filter[0] // 2)
        self.attn2 = AttentionBlock(nb_filter[1], nb_filter[2], nb_filter[1] // 2)
        self.attn3 = AttentionBlock(nb_filter[2], nb_filter[3], nb_filter[2] // 2)
        self.attn4 = AttentionBlock(nb_filter[3], nb_filter[4], nb_filter[3] // 2)

    def forward(self, input):
        # Encoder
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Attention Gates
        g1 = self.attn1(x0_0, x1_0)
        g2 = self.attn2(x1_0, x2_0)
        g3 = self.attn3(x2_0, x3_0)
        g4 = self.attn4(x3_0, x4_0)

        # Decoder with Attention
        x0_1 = self.conv0_1(torch.cat([g1, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([g2, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([g3, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([g4, self.up(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # Output
        output = None

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)

            output = [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)

        return output


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels, num_heads=8):
        super().__init__()
        
        # Ensure that inter_channels is a multiple of num_heads for MultiheadAttention
        self.inter_channels = inter_channels
        
        # Conv layers to process input and gating channels
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Multihead Attention layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=inter_channels, num_heads=num_heads, batch_first=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, g):
        B, C, H, W = x.size()

        # Apply convolutions to get query and key
        theta_x = self.theta(x).view(B, self.inter_channels, -1).permute(0, 2, 1)  # B x H*W x C
        phi_g = self.phi(g).view(B, self.inter_channels, -1).permute(0, 2, 1)  # B x H*W x C

        # Ensure dimensions match for attention
        attn_output, _ = self.multihead_attention(theta_x, phi_g, phi_g)  # Output: B x H*W x C

        # Reshape back to spatial dimensions
        new_H, new_W = self.theta(x).size(2), self.theta(x).size(3)  # Infer spatial size after theta
        attn_output = attn_output.permute(0, 2, 1).view(B, self.inter_channels, new_H, new_W)

        # Upsample and compute attention mask
        attn_output_upsampled = self.upsample(attn_output)
        psi = self.sigmoid(self.psi(attn_output_upsampled))

        # Apply attention mask
        return x * psi

class NestedUNetWithMultiheadAttention(NestedUNet):
    def __init__(self, num_classes, input_channels=1, deep_supervision=False):
        super().__init__(num_classes, input_channels, deep_supervision)

        nb_filter = [64, 128, 256, 512, 1024]

        # Use multi-head attention blocks for two layers
        self.attn2 = MultiheadAttentionBlock(nb_filter[1], nb_filter[2], nb_filter[1] // 2)
        self.attn4 = MultiheadAttentionBlock(nb_filter[3], nb_filter[4], nb_filter[3] // 2)

    def forward(self, input):
        # Encoder
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Apply Multihead Attention Gates at selected levels
        g2 = self.attn2(x1_0, x2_0)
        g4 = self.attn4(x3_0, x4_0)

        # Decoder with Attention
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([g2, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([g4, self.up(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # Output
        output = None

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)

            output = [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)

        return output
