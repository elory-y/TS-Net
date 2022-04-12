import torch
import math
import copy
import torch.nn as nn
from scipy import ndimage
import numpy as np
from os.path import join as pjoin
import torch.nn.functional as F

class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query/"
ATTENTION_K = "MultiHeadDotProductAttention_1/key/"
ATTENTION_V = "MultiHeadDotProductAttention_1/value/"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out/"

FC_0 = "MlpBlock_3/Dense_0/"
FC_1 = "MlpBlock_3/Dense_1/"
ATTENTION_NORM = "LayerNorm_0/"
MLP_NORM = "LayerNorm_2/"

def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)
        self.act_fn = nn.functional.gelu  # todo 3-11
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = 12
        self.attention_head_size = int(768 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(768, self.all_head_size)
        self.key = nn.Linear(768, self.all_head_size)
        self.value = nn.Linear(768, self.all_head_size)

        self.out = nn.Linear(768, 768)
        self.attn_dropout = nn.Dropout(0.0)
        self.proj_dropout = nn.Dropout(0.0)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # key_layer.transpose 交换维度
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))       # tensor的乘法
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None    
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Block(nn.Module):
    def __init__(self, vis):
        super(Block, self).__init__()
        self.hidden_size = 768
        self.attention_norm = nn.LayerNorm(768, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(768, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention(vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}/"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()      # todo ???
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, 'kernel')]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, 'kernel')]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, 'bias')]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, 'bias')]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Transformer(nn.Module):
    def __init__(self, vis):
        super(Transformer, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(768, eps=1e-6)
        for _ in range(6):
            layer = Block(vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Encoder(nn.Module):
    def __init__(self, vis):
        super(Encoder, self).__init__()

        self.encoder_1 = nn.Sequential(
            SingleConv(5, 32, kernel_size=3, stride=1, padding=1),
            SingleConv(32, 32, kernel_size=3, stride=1, padding=1))

        self.encoder_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(32, 64, kernel_size=3, stride=1, padding=1),
            SingleConv(64, 64, kernel_size=3, stride=1, padding=1))

        self.encoder_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(64, 128, kernel_size=3, stride=1, padding=1),
            SingleConv(128, 128, kernel_size=3, stride=1, padding=1))

        self.encoder_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(128, 256, kernel_size=3, stride=1, padding=1),
            SingleConv(256, 256, kernel_size=3, stride=1, padding=1))

        self.encoder_5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(256, 512, kernel_size=3, stride=1, padding=1),
            SingleConv(512, 512, kernel_size=3, stride=1, padding=1))

        self.patch_embeddings = nn.Conv2d(in_channels=512,
                                          out_channels=768,
                                          kernel_size=1,
                                          stride=1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 256, 768))
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.transformer = Transformer(vis)

    def forward(self, x):

        out_encoder_1 = self.encoder_1(x)

        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        out_encoder_5 = self.encoder_5(out_encoder_4)

        x = self.patch_embeddings(out_encoder_5)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        encoded, attn_weights = self.transformer(embeddings)
        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, encoded, attn_weights]


class FAM(nn.Module):
    def __init__(self, in_chanl, out_chanl):
        super(FAM, self).__init__()
        self.down_h = nn.Conv2d(in_chanl, out_chanl, 1, bias=False)
        self.down_l = nn.Conv2d(out_chanl, out_chanl, 1, bias=False)
        self.flow_make = nn.Conv2d(out_chanl * 2, 2, kernel_size=3, padding=1, bias=False)
        self.feat_fuse = nn.Conv2d(in_chanl, out_chanl, kernel_size=1, bias=False)

    def forward(self, high_feature, low_feature):
        
        return 

    def flow_warp(self, input, flow, size):
   
        return output


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_more = SingleConv(768, 512, kernel_size=3, stride=1, padding=1)

        self.FAM_4 = FAM(512, 256)
        self.upconv_4_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=True)
        self.decoder_conv_4_1 = nn.Sequential(
            SingleConv(512, 256, kernel_size=3, stride=1, padding=1),
            SingleConv(256, 256, kernel_size=3, stride=1, padding=1))

        self.FAM_3 = FAM(256, 128)
        self.upconv_3_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=True)
        self.decoder_conv_3_1 = nn.Sequential(
            SingleConv(256, 128, kernel_size=3, stride=1, padding=1),
            SingleConv(128, 128, kernel_size=3, stride=1, padding=1))

        self.FAM_2 = FAM(128,64)
        self.upconv_2_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=True)
        self.decoder_conv_2_1 = nn.Sequential(
            SingleConv(128, 64, kernel_size=3, stride=1, padding=1),
            SingleConv(64, 64, kernel_size=3, stride=1, padding=1))

        self.FAM_1 = FAM(64, 32)
        self.upconv_1_1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, bias=True)
        self.decoder_conv_1_1 = nn.Sequential(
            SingleConv(64, 32, kernel_size=3, stride=1, padding=1),
            SingleConv(32, 32, kernel_size=3, stride=1, padding=1))

        self.conv_out = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, padding=0, bias=True)
        )


    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, hidden_states, attn_weights = out_encoder
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        out_transf = hidden_states.permute(0, 2, 1)
        out_transf = out_transf.contiguous().view(B, hidden, h, w)
        out_transf = self.conv_more(out_transf)
        out_FAM_4 = self.FAM_4(out_transf, out_encoder_4)
        out_decoder_4_1 = self.decoder_conv_4_1(torch.cat((self.upconv_4_1(out_transf), out_FAM_4), dim=1))

        out_FAM_3 = self.FAM_3(out_decoder_4_1, out_encoder_3)
        out_decoder_3_1 = self.decoder_conv_3_1(torch.cat((self.upconv_3_1(out_decoder_4_1), out_FAM_3), dim=1))

        out_FAM_2 = self.FAM_2(out_decoder_3_1, out_encoder_2)
        out_decoder_2_1 = self.decoder_conv_2_1(torch.cat((self.upconv_2_1(out_decoder_3_1), out_FAM_2), dim=1))

        out_FAM_1 = self.FAM_1(out_decoder_2_1, out_encoder_1)
        out_decoder_1_1 = self.decoder_conv_1_1(torch.cat((self.upconv_1_1(out_decoder_2_1), out_FAM_1), dim=1))

        output = self.conv_out(out_decoder_1_1)
        return [output]


class Model(nn.Module):
    def __init__(self, vis):
        super(Model, self).__init__()
        self.encoder = Encoder(vis)
        self.decoder = Decoder()

        self.initialize()

    @staticmethod
    def init_conv_deconv_BN(modules):
        for m in modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def initialize(self):
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_deconv_BN(self.decoder.modules)
        self.load_from(weights=np.load('../../PretrainedModels/R50+ViT-B_16.npz'))
        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_deconv_BN(self.encoder.modules)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)

        return out_decoder

    def load_from(self, weights):           # todo 4-1
        with torch.no_grad():

            res_weight = weights
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.encoder.position_embeddings

            if posemb.size() == posemb_new.size():
                self.encoder.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.encoder.position_embeddings.copy_(posemb)

            else:
                # logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if True:
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.encoder.position_embeddings.copy_(np2th(posemb))

if __name__=='__main__':
    x = torch.rand((1, 5, 256, 256))
    net = Model(vis=False)
    y = net(x)
