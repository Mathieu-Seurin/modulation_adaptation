import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from .film_utils import ResidualBlock, FiLMedResBlock, init_modules
from .gpu_utils import FloatTensor, USE_CUDA
from .fusion_utils import choose_fusing, lstm_last_step, lstm_whole_seq, TextAttention,\
    ConvPoolReducingLayer, PoolReducingLayer, LinearReducingLayer

from torch.nn import CrossEntropyLoss
import logging
import numpy as np

def count_good_prediction(yhat,y):
    model_pred = torch.max(yhat, 1)[1].cpu().data.numpy()
    y = y.cpu().data.numpy()
    return np.equal(model_pred,y).sum()

def compute_accuracy(yhat, y):
    return count_good_prediction(yhat=yhat,y=y) / len(yhat)


class ClfModel(nn.Module):
    def __init__(self, config, n_class, input_info):

        super(ClfModel, self).__init__()

        self.forward_model = FilmedNet(config=config,
                                       n_class=n_class,
                                       input_info=input_info)

        # Init Loss
        self.loss_func = CrossEntropyLoss()

        if USE_CUDA :
            self.forward_model.cuda()
            self.loss_func.cuda()

        init_modules(self.modules())


    def forward(self, x):
        return self.forward_model(x)

    def optimize(self, x, y):

        yhat = self.forward_model(x)
        loss = self.loss_func(yhat, y)

        self.forward_model.optimizer.zero_grad()
        loss.backward()

        # for param in self.forward_model.parameters():
        #     #logging.debug(param.grad.data.sum())
        #     param.grad.data.clamp_(-1., 1.)

        self.forward_model.optimizer.step()

        return loss.data.cpu(), yhat

    def new_task(self, num_task):

        raise NotImplementedError("Not available yet")

        if num_task == 0:
            self.forward_model.use_film = False

        # todo : change hardcoded, do scheduler of something like this ?
        if num_task > 0:

            # todo : check resnet
            self.forward_model.use_film = True
            for param_group in self.forward_model.optimizer.param_groups:
                if param_group['name'] == "base_net_params":
                    param_group['lr'] = self.forward_model.after_lr
                elif param_group['name'] == "film_params":
                    param_group['lr'] = self.forward_model.default_lr



class MultiHopFilmGen(nn.Module):

    def __init__(self, config, n_block_to_modulate, n_feature_map_per_block, text_size, vision_size=None):
        super(MultiHopFilmGen, self).__init__()
        assert vision_size is not None, "For FiLM with feedback loop, need size of visual features"

        self.text_size = text_size
        self.vision_size = vision_size
        self.vision_after_reduce_size_mlp = config["vision_reducing_size_mlp"]
        self.film_gen_hidden_size = config["film_gen_hidden_size"]
        self.use_feedback = config["use_feedback"]

        self.n_feature_map_per_block = n_feature_map_per_block

        self.attention = TextAttention(hidden_mlp_size=config["film_attention_size_hidden"],
                                       text_size=text_size)

        if self.use_feedback:
            if config["vision_reducing_method"] == "mlp" :
                vision_size_flatten = vision_size[1]*vision_size[2]*vision_size[3] # just flatten the input
                self.vision_reducer_layer = LinearReducingLayer(vision_size_flatten=vision_size_flatten,
                                                                output_size=self.vision_after_reduce_size_mlp)
            elif config["vision_reducing_method"] == "conv" :
                self.vision_reducer_layer = ConvPoolReducingLayer(vision_size[1])
            elif config["vision_reducing_method"] == "pool" :
                self.vision_reducer_layer = PoolReducingLayer()
            else:
                raise NotImplementedError("Wrong vision reducing method : {}".format(config["vision_reducing_method"]))

            vision_after_reduce_size = self.compute_reduction_size()[1] # batch_size, n_features
        else:
            vision_after_reduce_size = 0

        self.film_gen_hidden = nn.Linear(self.text_size + vision_after_reduce_size, self.film_gen_hidden_size)
        self.film_gen_last_layer = nn.Linear(self.film_gen_hidden_size, n_feature_map_per_block * 2)
        # for every feature_map, you generate a beta and a gamma, to do : feature_map*gamma + beta
        # So, for every feature_map, 2 parameters are generated


    def forward(self, text, first_layer, vision=None):
        """
        Common interface for all Film Generator
        first_layer indicate that you calling film generator for the first time (needed for init etc...)
        """
        batch_size = text.size(1)

        # todo : learn init from vision ??
        # if first layer, reset ht to ones only
        if first_layer:
            self.ht = Variable(torch.ones(batch_size, self.text_size).type(FloatTensor))

        # Compute text features
        text_vec = self.attention(text_seq=text, previous_hidden=self.ht)
        # todo layer norm ? not available on 0.3.0
        self.ht = text_vec


        # Compute feedback loop and fuse
        if self.use_feedback:
            vision_feat_reduced = self.vision_reducer_layer(vision)
            film_gen_input = torch.cat((vision_feat_reduced, text_vec), dim=1)
        else:
            film_gen_input = text_vec

        # Generate film parameters
        hidden_film_gen_activ = F.relu(self.film_gen_hidden(film_gen_input))
        gammas_betas = self.film_gen_last_layer(hidden_film_gen_activ)

        return gammas_betas

    def compute_reduction_size(self):

        tmp = Variable(torch.ones(self.vision_size))
        tmp_out = self.vision_reducer_layer(tmp)
        return tmp_out.size()


class TextEmbedEncoder(nn.Module):
    def __init__(self,config, need_all_ht, text_size, vocab_size):
        super(TextEmbedEncoder, self).__init__()

        # Dealing with text / second input
        embedding_dim = config["embedding_dim"]
        self.rnn_hidden_size = config["rnn_hidden_size"]
        self.text_size = text_size

        # simple film need just the last state, multi-hop need all ht.
        self.return_all_ht = need_all_ht

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim
                                      )

        self.rnn = nn.GRU(num_layers=config["n_rnn_layers"],
                          input_size=embedding_dim,
                          hidden_size=self.rnn_hidden_size,
                          batch_first=True,
                          bidirectional=False)

    def forward(self, text):

        text = self.embedding(text)
        all_ht, ht = self.rnn(text)

        if self.return_all_ht:
            raise NotImplementedError("Need test")
            return all_ht
        else:
            return all_ht[:, -1]



class SimpleFilmGen(nn.Module):

    def __init__(self, config, n_block_to_modulate, n_feature_map_per_block, input_size):
        super(SimpleFilmGen, self).__init__()

        self.n_block_to_modulate = n_block_to_modulate

        self.n_feature_map_per_block = n_feature_map_per_block
        self.n_features_to_modulate = self.n_block_to_modulate * self.n_feature_map_per_block

        #self.film_gen_hidden = nn.Linear(input_size, self.film_gen_hidden_size)
        self.film_gen_last_layer = nn.Linear(input_size, self.n_features_to_modulate * 2)
        # for every feature_map, you generate a beta and a gamma, to do : feature_map*gamma + beta
        # So, for every feature_map, 2 parameters are generated

    def forward(self, text, first_layer, vision=None):
        """
        Common interface for all Film Generator
        first_layer indicate that you calling film generator for the first time (needed for init etc...)
        """
        if first_layer:
            self.num_layer_count = 0
            #hidden_film_gen_activ = F.relu(self.film_gen_hidden(text))
            self.gammas_betas = self.film_gen_last_layer(text)

        gamma_beta_id = slice(self.n_feature_map_per_block * self.num_layer_count * 2,
                              self.n_feature_map_per_block * (self.num_layer_count + 1) * 2)

        self.num_layer_count += 1

        return self.gammas_betas[:, gamma_beta_id]


class FilmedNet(nn.Module):
    def __init__(self, config, n_class, input_info):
        super(FilmedNet, self).__init__()

        # General params
        self.default_lr = config["base_learning_rate"]

        # Resblock params
        self.n_regular_block = config["n_regular_block"]
        self.n_modulated_block = config["n_modulated_block"]
        self.input_shape = input_info['vision_shape']
        self.n_channel_input = self.input_shape[1]
        self.n_feature_map_per_block = config["n_feat_map_max"]

        self.regular_blocks = nn.ModuleList()
        self.modulated_blocks = nn.ModuleList()


        # Head params
        self.n_channel_head = config["head_channel"]
        self.kernel_size_head = config["head_kernel"]

        # If use attention as fusing : no head/pooling
        self.use_attention_as_fusing = config['fusing_method'] == 'attention'

        # Film type
        self.use_multihop = False
        self.use_film = config["use_film"]
        if self.use_film:
            self.film_gen_type = config["film_gen_params"]["film_type"]
            self.use_multihop = config["film_gen_params"]["film_type"] == "multi_hop"

        # FC params
        self.fc_n_hidden = config['fc_n_hidden']
        self.fc_dropout = config["fc_dropout"]
        self.n_class = n_class

        # Second modality (aka text for clevr)
        self.second_modality_shape = input_info['second_modality_shape']

        if input_info["second_modality_type"] == "text":
            self.text_embed_encode = TextEmbedEncoder(config['text_encoding'],
                                                      need_all_ht=self.use_multihop,
                                                      text_size=self.second_modality_shape,
                                                      vocab_size=input_info['vocab_size'])
            # todo : deal with multiple ht
            self.encoded_text_size = self.text_embed_encode.rnn_hidden_size
        else:
            raise NotImplementedError("Not tested")
            self.text_embed_encode = lambda x:x
            self.encoded_text_size = 0


        # Learned features extractor
        self.n_feature_extactor_channel = config["features_extractor"]["n_channel"]
        self.feature_extactor = nn.Sequential()
        shape_input_to_next_block = self.n_channel_input

        for layer in range(config["features_extractor"]["n_layer"]):
            self.feature_extactor.add_module("conv"+str(layer), nn.Conv2d(in_channels=shape_input_to_next_block,
                                                                          out_channels=self.n_feature_extactor_channel,
                                                                          kernel_size=4, stride=2, padding=1))

            self.feature_extactor.add_module("bn"+str(layer), nn.BatchNorm2d(self.n_feature_extactor_channel))
            self.feature_extactor.add_module("relu"+str(layer), nn.ReLU())

            shape_input_to_next_block = self.n_feature_extactor_channel


        # RESBLOCK BUILDING
        #==================
        # Create resblock, not modulated by FiLM
        for regular_resblock_num in range(self.n_regular_block):

            current_regular_resblock = ResidualBlock(in_dim=shape_input_to_next_block,
                                                     out_dim=self.n_feature_map_per_block,
                                                     with_residual=True,
                                                     with_batchnorm=False)
            shape_input_to_next_block = self.n_feature_map_per_block

            self.regular_blocks.append(current_regular_resblock)



        # Create FiLM-ed resblock
        for modulated_block_num in range(self.n_modulated_block):
            current_modulated_resblock = FiLMedResBlock(in_dim=shape_input_to_next_block,
                                                        out_dim=self.n_feature_map_per_block,
                                                        with_residual=True,
                                                        with_batchnorm=True) #with_cond=[True], dropout=self.resblock_dropout)
            shape_input_to_next_block = self.n_feature_map_per_block

            self.modulated_blocks.append(current_modulated_resblock)


        # head
        if self.kernel_size_head != 0 and not self.use_attention_as_fusing:
            self.head_conv = nn.Conv2d(in_channels=shape_input_to_next_block,
                                   out_channels=self.n_channel_head,
                                   kernel_size=self.kernel_size_head)
        else:
            self.head_conv = lambda x:x

        fc_input_size, intermediate_conv_size = self.compute_conv_size()
        self.fc1 = nn.Linear(in_features=fc_input_size, out_features=self.fc_n_hidden)
        self.fc2 = nn.Linear(in_features=self.fc_n_hidden, out_features=self.n_class)


        if self.use_film:

            if self.film_gen_type == "simple":
                self.film_gen = SimpleFilmGen(config=config['film_gen_params'],
                                              n_block_to_modulate=self.n_modulated_block,
                                              n_feature_map_per_block=self.n_feature_map_per_block,
                                              input_size=self.encoded_text_size
                                              )

            elif self.film_gen_type == "multi_hop":
                self.film_gen = MultiHopFilmGen(config=config['film_gen_params'],
                                                n_block_to_modulate=self.n_modulated_block,
                                                n_feature_map_per_block=self.n_feature_map_per_block,
                                                text_size=self.second_modality_shape[1],
                                                vision_size=intermediate_conv_size)
            else:
                raise NotImplementedError("Wrong Film generator type : given '{}'".format(self.film_gen_type))


        # Optimizer
        optimizer = config['optimizer'].lower()

        optim_config = [
            {'params': self.parameters(), 'weight_decay': config['default_w_decay'], 'name': "base_net_params"}, # Default config
        ]

        # optim_config = [
        #     {'params': self.get_all_params_except_film(), 'weight_decay': config['default_w_decay'], 'name': "base_net_params"}, # Default config
        # ]
        #
        # if self.use_film:
        #     optim_config.append({'params': self.film_gen.parameters(), 'weight_decay': config['FiLM_decay'], 'name': "film_params"})  # Film gen parameters
        #     assert len([i for i in optim_config[1]['params']]) + len([i for i in optim_config[0]['params']]) == len([i for i in self.parameters()])

        if optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(optim_config, lr=self.default_lr)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(optim_config, lr=self.default_lr)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(optim_config, lr=self.default_lr)
        else:
            assert False, 'Optimizer not recognized'

    def forward(self, x):

        x, second_mod = x['vision'], x['second_modality']

        second_mod = self.text_embed_encode(second_mod)

        x = self.compute_conv(x, second_mod, still_building_model=False)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.fc_dropout, training=self.training)
        x = self.fc2(x)
        return x

    def compute_conv_size(self):

        # Don't convert it to cuda because the model is not yet on GPU (because you're still defining the model here ;)
        tmp = Variable(torch.zeros(*self.input_shape))
        return self.compute_conv(tmp, still_building_model=True).size(1), self.intermediate_conv_size

    def compute_conv(self, x, text_state=None, still_building_model=False):
        """
        :param x: vision input with batch dimension first
        :param text_state: all hidden states of the lstm encoder
        :param still_building_model: needed if you use this function just to get the output size
        :return: return visual features, modulated if FiLM
        """

        if self.use_film:
            if not still_building_model:
                assert text_state is not None, "if you use film, need to provide text as input too"

        batch_size = x.size(0)

        x = self.feature_extactor(x)

        # Regular resblock, easy
        for i,regular_resblock in enumerate(self.regular_blocks):
            x = regular_resblock.forward(x)
            self.intermediate_conv_size = x.size()


        #Modulated block : first compute FiLM weights, then send them to the resblock layers
        for i,modulated_resblock in enumerate(self.modulated_blocks):

            if still_building_model or not self.use_film :
                # Gammas = all zeros   Betas = all zeros
                gammas = Variable(torch.zeros(batch_size, self.n_feature_map_per_block).type_as(x.data))
                betas = Variable(torch.zeros_like(gammas.data).type_as(x.data))
            else: # use film
                gammas_betas = self.film_gen.forward(text_state, first_layer= i==0, vision=x)
                assert gammas_betas.size(1)%2 == 0, "Problem, more gammas than betas (or vice versa)"
                middle = gammas_betas.size(1)//2
                gammas = gammas_betas[:,:middle]
                betas = gammas_betas[:, middle:]

            x = modulated_resblock.forward(x, gammas=gammas, betas=betas)
            self.intermediate_conv_size = x.size()

        if not self.use_attention_as_fusing:
            x = self.head_conv(x)
            x = F.max_pool2d(x, kernel_size=x.size()[2:])

        return x.view(batch_size, -1)

    def get_all_params_except_film(self):

        params = []
        for name, param in self.named_parameters():
            if "film_gen" not in name:
                params.append(param)

        return params



if __name__ == "__main__":
    pass
