import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class Mean_aggregator(nn.Module):
    def __init__(self, add_self, sample_num):
        super(Mean_aggregator, self).__init__()
        self.add_self = add_self
        self.sample_num = sample_num

    def forward(self, node_features, nodes, adj_list):
        if self.add_self:
            adj_list = adj_list + torch.eye(adj_list.size(-1)).unsqueeze(0).to(adj_list.device)

        neighber_sum = adj_list.sum(-1, keepdim=True)
        mask = adj_list / neighber_sum
        mask = torch.where(torch.isnan(mask), torch.full_like(mask, 0.), mask)
        output = torch.bmm(mask, node_features)
        return output


class Attention_aggregator(nn.Module):
    def __init__(self, add_self, sample_num):
        super(Attention_aggregator, self).__init__()
        self.add_self = add_self
        self.sample_num = sample_num

    def forward(self, node_features, nodes, adj_list):
        if self.add_self:
            adj_list = adj_list + torch.eye(adj_list.size(-1)).unsqueeze(0).to(adj_list.device)

        attention = torch.matmul(node_features, node_features.permute([0, 2, 1]))
        attention[adj_list == 0] = -9999999
        masked_attention = F.softmax(attention, dim=-1)
        output = torch.matmul(masked_attention, node_features)
        return output


class LSTM_aggregator(nn.Module):
    def __init__(self, add_self, sample_num, input_dim, output_dim, agent_num):
        super(LSTM_aggregator, self).__init__()
        self.add_self = add_self
        self.sample_num = sample_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.agent_num = agent_num

        self.lstm = nn.GRU(self.input_dim, self.output_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.linear_mapping = nn.Linear(self.output_dim * 2, self.agent_num)

    def forward(self, node_features, nodes, adj_list):
        if self.add_self:
            adj_list = adj_list + torch.eye(adj_list.size(-1)).unsqueeze(0).to(adj_list.device)

        attention = self.lstm(node_features)[0]
        attention_scores = self.linear_mapping(attention)
        attention_scores[adj_list == 0] = -9999999
        masked_attention_scores = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(masked_attention_scores, attention)
        return output


class Sum_aggregator(nn.Module):
    def __init__(self, add_self, sample_num):
        super(Sum_aggregator, self).__init__()
        self.add_self = add_self
        self.sample_num = sample_num

    def forward(self, node_features, nodes, adj_list):
        if self.add_self:
            adj_list = adj_list + torch.eye(adj_list.size(-1)).unsqueeze(0).to(adj_list.device)

        output = torch.bmm(adj_list, node_features)
        return output


class La_aggregator(nn.Module):
    def __init__(self, add_self, sample_num, input_dim, agent_num):
        super(La_aggregator, self).__init__()
        self.add_self = add_self
        self.sample_num = sample_num
        self.input_dim = input_dim
        self.agent_num = agent_num

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.input_dim),
        )

    def forward(self, node_features, nodes, adj_list):
        if self.add_self:
            adj_list = adj_list + torch.eye(adj_list.size(-1)).unsqueeze(0).to(adj_list.device)

        indiv_dim = node_features.size(-1)
        node_features_repeat = node_features.repeat([1, self.agent_num, 1])
        node_features_alter_repeat = node_features.repeat_interleave(repeats=self.agent_num, dim=1)
        atten_input = torch.cat([node_features_repeat, node_features_alter_repeat], dim=-1).view(-1, self.agent_num, self.agent_num, indiv_dim * 2)

        mask = self.mlp(atten_input)
        mask = mask * adj_list.unsqueeze(-1)
        mask = F.sigmoid(mask)

        output = (mask * node_features.unsqueeze(1)).sum(-2)
        return output

class Max_aggregator(nn.Module):
    def __init__(self, add_self, sample_num):
        super(Max_aggregator, self).__init__()
        self.add_self = add_self
        self.sample_num = sample_num

    def forward(self, node_features, nodes, adj_list):
        if self.add_self:
            adj_list = adj_list + torch.eye(adj_list.size(-1)).unsqueeze(0).to(adj_list.device)

        max_features = []
        for r in range(adj_list.size(0)):
            max_features.append(node_features.max(0)[0])
        max_features = torch.stack(max_features, dim=0)
        return max_features


class Encoder(nn.Module):
    def __init__(self, aggregator, feature_dim, embed_dim, concat):
        super(Encoder, self).__init__()
        self.aggregator = aggregator
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.concat = concat

        self.encoder_layer = nn.Linear(self.feature_dim * 2 if self.concat or isinstance(self.aggregator, LSTM_aggregator) else self.feature_dim, self.embed_dim, bias=False)

    def forward(self, node_features, nodes, adj_list):
        neigh_feats = self.aggregator.forward(node_features, nodes, adj_list)
        self_feats = node_features
        if self.concat:
            combined = torch.cat([self_feats, neigh_feats], dim=-1)
        else:
            combined = neigh_feats
        encoder_weight = F.relu(self.encoder_layer(combined))
        return encoder_weight


class MganMixer(nn.Module):
    def __init__(self, args):
        super(MganMixer, self).__init__()
        self.args = args
        self.aggregator_args = self.args.aggregator_args
        self.add_self = args.add_self
        self.concat = args.concat
        self.sample_num = args.sample_num
        self.hidden_dim = args.hidden_dim
        self.hyper_hidden_dim = args.hyper_hidden_dim
        self.head_num = args.head_num

        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.indiv_u_dim = int(np.prod(args.observation_shape))

        if self.aggregator_args['method'] == 'mean':
            self.aggregator_1 = Mean_aggregator(self.add_self, self.sample_num)
            self.aggregator_2 = Mean_aggregator(self.add_self, self.sample_num)
        elif self.aggregator_args['method'] == 'max':
            self.aggregator_1 = Max_aggregator(self.add_self, self.sample_num)
            self.aggregator_2 = Max_aggregator(self.add_self, self.sample_num)
        elif self.aggregator_args['method'] == 'atten':
            self.aggregator_1 = Attention_aggregator(self.add_self, self.sample_num)
            self.aggregator_2 = Attention_aggregator(self.add_self, self.sample_num)
        elif self.aggregator_args['method'] == 'lstm':
            self.aggregator_1 = LSTM_aggregator(self.add_self, self.sample_num, self.indiv_u_dim, self.indiv_u_dim, self.n_agents)
            self.aggregator_2 = LSTM_aggregator(self.add_self, self.sample_num, self.hidden_dim * 2 if self.concat else self.hidden_dim, self.hidden_dim, self.n_agents)
        elif self.aggregator_args['method'] == 'sum':
            self.aggregator_1 = Sum_aggregator(self.add_self, self.sample_num)
            self.aggregator_2 = Sum_aggregator(self.add_self, self.sample_num)
        elif self.aggregator_args['method'] == 'la':
            self.aggregator_1 = La_aggregator(self.add_self, self.sample_num, self.indiv_u_dim, self.n_agents)
            self.aggregator_2 = La_aggregator(self.add_self, self.sample_num, self.hidden_dim, self.n_agents)


        self.encoder_1 = nn.ModuleList([Encoder(self.aggregator_1, self.indiv_u_dim, self.hidden_dim, self.concat) for _ in range(self.head_num)])
        self.encoder_2 = nn.ModuleList([Encoder(self.aggregator_2, self.hidden_dim, self.hidden_dim, self.concat) for _ in range(self.head_num)])

        self.output_layer = nn.Linear(self.hidden_dim, 1)

        self.hyper_weight_layer = nn.Sequential(
            nn.Linear(self.state_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.head_num * 1)
        )
        self.hyper_const_layer = nn.Sequential(
            nn.Linear(self.state_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, 1)
        )


    def forward(self, agent_qs, states, indiv_us, adj_list):
        bs = agent_qs.size(0)
        sl = agent_qs.size(1)

        if adj_list is None:
            adj_list = torch.ones([bs, sl, self.n_agents, self.n_agents])
            adj_list = adj_list - torch.eye(self.n_agents).unsqueeze(0).unsqueeze(0)
            adj_list = adj_list.to(states.device)
        agent_qs = agent_qs.view(-1, agent_qs.size(-1))
        states = states.reshape(-1, states.size(-1))
        indiv_us = indiv_us.reshape(-1, indiv_us.size(-2), indiv_us.size(-1))
        adj_list = adj_list.reshape(-1, adj_list.size(-2), adj_list.size(-1))
        node_features = indiv_us
        nodes = torch.LongTensor(list(range(self.n_agents)))
        enc_outputs = []
        for h in range(self.head_num):
            enc_output = self.encoder_2[h](self.encoder_1[h].forward(node_features, nodes, adj_list), nodes, adj_list)
            enc_outputs.append(enc_output)
        enc_outputs = torch.stack(enc_outputs, dim=-2)
        output_weight = self.output_layer(enc_outputs).squeeze()
        output_weight = F.softmax(output_weight, dim=-2)
        qs_tot = torch.bmm(agent_qs.unsqueeze(1), output_weight)

        hyper_weight = torch.abs(self.hyper_weight_layer(states).view(-1, self.head_num, 1))
        hyper_const = self.hyper_const_layer(states).view(-1, 1, 1)
        q_tot = torch.bmm(qs_tot, hyper_weight) + hyper_const
        return q_tot.view(bs, sl, 1)