# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from signal import pthread_kill
import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, vector_state_shape, feature_dim, num_layers=2, num_filters=32, stride=None):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0] * 3, num_filters, 3, stride=2)]
        ) # obs_shape[0] * 3 (3) is cfg['k_frame']
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = {2: 39, 4: 35, 6: 31}[num_layers]
        self.fcat = nn.Linear(num_filters * out_dim * out_dim + vector_state_shape[0] * 3, self.feature_dim, nn.ReLU)
        # vector_state_shape[0] * 3 (3) is cfg['k_frame']
        self.fc = nn.Linear(self.feature_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs
        
        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, vector_state, detach=False, pre_ratio=False):
        self.outputs['vector_state'] = vector_state
        h = self.forward_conv(obs)
        vector_state = vector_state.reshape(-1, 9)
        h = torch.cat((h, vector_state), dim=1) #cat map and goal
        #print(h.shape)
        if detach:
            h = h.detach()

        h = self.fcat(h)
        self.outputs['fcat'] = h
        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        out = self.ln(h_fc)

        self.outputs['ln'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])
    # def copy_conv_weights_from(self, source):
    #     """Tie convolutional layers"""
    #     # only tie conv layers
    #     for i in range(self.num_layers):
    #         tie_weights(src=source.convs[i], trg=self.convs[i])
    #     tie_weights(src=source.fc, trg=self.fc)
    #     tie_weights(src=source.ln, trg=self.ln)
    #     self.ratio = source.ratio


    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fcat', self.fcat, step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class PixelEncoderCarla096(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, rew_length, num_layers=2, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=stride))

        out_dims = 100 #100 # 16 # if defaults change, adjust this as needed
        self.fc = nn.Linear(num_filters * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.rew_length = rew_length
        self.dyn_length = feature_dim - rew_length
        self.ratio =  nn.Parameter(torch.tensor([0., 0.]))

        self.outputs = dict()


class PixelEncoderCarla098(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, rew_length, num_layers=4, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()
        print("PixelEncoderCarla098")

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        # self.convs.append(nn.Conv2d(obs_shape[0], 64, 5, stride=2))
        # self.convs.append(nn.Conv2d(64, 128, 3, stride=2))
        # self.convs.append(nn.Conv2d(128, 256, 3, stride=2))
        # self.convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.convs.append(nn.Conv2d(obs_shape[0], 64, 5, stride=2))
        self.convs.append(nn.Conv2d(64, 64, 3, stride=2))
        self.convs.append(nn.Conv2d(64, 64, 3, stride=2))
        self.convs.append(nn.Conv2d(64, 64, 3, stride=2))

        # out_dims = 56  # 3 cameras
        # out_dims = 100  # 5 cameras
        out_dims = 16
        self.fc = nn.Linear(64 * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.rew_length = rew_length
        self.dyn_length = feature_dim - rew_length
        self.ratio =  nn.Parameter(torch.tensor([0., 0.]))

        self.outputs = dict()


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder,
                       'pixelCarla096': PixelEncoderCarla096,
                       'pixelCarla098': PixelEncoderCarla098,
                       'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, vector_state_shape, feature_dim, num_layers, num_filters, stride
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, vector_state_shape, feature_dim, num_layers, num_filters, stride
    )
