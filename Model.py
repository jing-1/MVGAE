import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from BaseModel import BaseModel
from torch.autograd import Variable

EPS = 1e-15
MAX_LOGVAR = 10


class GCN(torch.nn.Module):
    def __init__(self, device, features, edge_index, batch_size, num_user, num_item, dim_id, aggr_mode, concate,
                 num_layer, dim_latent=None):
        super(GCN, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.features = features
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer

        if self.dim_latent:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).to(
                self.device)
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_id, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        else:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).to(
                self.device)
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_id, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

        self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_3.weight)
        self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        self.g_layer3 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

        self.conv_embed_4 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_4.weight)
        self.linear_layer4 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer4.weight)
        self.g_layer4 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)
        nn.init.xavier_normal_(self.g_layer4.weight)
        self.conv_embed_5 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_5.weight)
        self.linear_layer5 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer5.weight)
        self.g_layer5 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)
        nn.init.xavier_normal_(self.g_layer5.weight)

    def forward(self):
        temp_features = self.MLP(self.features) if self.dim_latent else self.features
        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x).to(self.device)

        if self.num_layer > 0:
            h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))
            x_hat = F.leaky_relu(self.linear_layer1(x))
            x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
                self.g_layer1(h))
            del x_hat
            del h

        if self.num_layer > 1:
            h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))
            x_hat = F.leaky_relu(self.linear_layer2(x))
            x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
                self.g_layer2(h))
            del h
            del x_hat
            
        if self.num_layer > 2:
            h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))
            x_hat = F.leaky_relu(self.linear_layer3(x))
            x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
                self.g_layer3(h))
            del h
            del x_hat

        mu = F.leaky_relu(self.conv_embed_4(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer4(x))
        mu = self.g_layer4(torch.cat((mu, x_hat), dim=1)) if self.concate else self.g_layer4(mu) + x_hat
        del x_hat

        logvar = F.leaky_relu(self.conv_embed_5(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer5(x))
        logvar = self.g_layer5(torch.cat((logvar, x_hat), dim=1)) if self.concate else self.g_layer5(logvar) + x_hat
        del x_hat
        return mu, logvar



class MVGAE(torch.nn.Module):
    def __init__(self, device, dataset, features, edge_index, batch_size, num_user, num_item, aggr_mode, concate,
                 num_layer, dim_x):
        super(MVGAE, self).__init__()
        self.experts = ProductOfExperts()
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.dim_x = dim_x
        self.collaborative = nn.init.xavier_normal_(torch.rand((num_item, 64), requires_grad=True)).to(self.device)

        self.edge_index = torch.tensor(edge_index).t().contiguous().to(device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        v_feat, t_feat = features
        self.v_feat = torch.tensor(v_feat, dtype=torch.float).to(device)
        self.t_feat = torch.tensor(t_feat, dtype=torch.float).to(device)
        del v_feat
        del t_feat

        self.v_gcn = GCN(self.device, self.v_feat, self.edge_index, batch_size, num_user, num_item, dim_x,
                         self.aggr_mode, self.concate, num_layer=num_layer, dim_latent=128)  # 256)
        self.t_gcn = GCN(self.device, self.t_feat, self.edge_index, batch_size, num_user, num_item, dim_x,
                         self.aggr_mode, self.concate, num_layer=num_layer, dim_latent=128)
        self.c_gcn = GCN(self.device, self.collaborative, self.edge_index, batch_size, num_user, num_item, dim_x,
                         self.aggr_mode, self.concate, num_layer=num_layer, dim_latent=128)
        self.result_embed = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_x))).to(device)

    def reparametrize(self, mu, logvar):
        logvar = logvar.clamp(max=MAX_LOGVAR)
        if self.training:
            return mu + torch.randn_like(logvar) * 0.1 * torch.exp(logvar.mul(0.5))
        else:
            return mu

    def dot_product_decode_neg(self, z, user, neg_items, sigmoid=True):
        # multiple negs, for comparison with MAML
        users = torch.unsqueeze(user, 1)
        neg_items = neg_items
        re_users = users.repeat(1, neg_items.size(1))

        neg_values = torch.sum(z[re_users] * z[neg_items], -1)
        max_neg_value = torch.max(neg_values, dim=-1).values
        return torch.sigmoid(max_neg_value) if sigmoid else max_neg_value

    def dot_product_decode(self, z, edge_index, sigmoid=True):
        value = torch.sum(z[edge_index[0]] * z[edge_index[1]], dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward(self):
        v_mu, v_logvar = self.v_gcn()
        t_mu, t_logvar = self.t_gcn()
        c_mu, c_logvar = self.c_gcn()
        self.v_logvar = v_logvar
        self.t_logvar = t_logvar
        self.v_mu = v_mu
        self.t_mu = t_mu
        mu = torch.stack([v_mu, t_mu], dim=0)
        logvar = torch.stack([v_logvar, t_logvar], dim=0)

        pd_mu, pd_logvar, self.pd_var = self.experts(mu, logvar)
        del mu
        del logvar

        mu = torch.stack([pd_mu, c_mu], dim=0)
        logvar = torch.stack([pd_logvar, c_logvar], dim=0)

        pd_mu, pd_logvar, self.pd_var = self.experts(mu, logvar)
        del mu
        del logvar
        z = self.reparametrize(pd_mu, pd_logvar)

        # for more sparse dataset like amazon, use signoid to regulization. for alishop,dont use sigmoid for better results
        if 'amazon' in self.dataset:
            self.result_embed = torch.sigmoid(pd_mu)
        else:
            self.result_embed = pd_mu
        return pd_mu, pd_logvar, z, v_mu, v_logvar, t_mu, t_logvar, c_mu, c_logvar

    def recon_loss(self, z, pos_edge_index, user, neg_items):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """
        # for more sparse dataset like amazon, use signoid to regulization. for alishop,dont use sigmoid for better results
        if 'amazon' in self.dataset:
            z = torch.sigmoid(z)

        pos_scores = self.dot_product_decode(z, pos_edge_index, sigmoid=False)
        neg_scores = self.dot_product_decode_neg(z, user, neg_items, sigmoid=False)
        loss = -torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        return loss

    def kl_loss(self, mu, logvar):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logvar`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logvar (Tensor, optional): The latent space for
                :math:`\log\sigma^2`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        logvar = logvar.clamp(max=MAX_LOGVAR)
        return -0.5 * torch.mean(
            torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))

    def loss(self, data):
        user, pos_items, neg_items = data
        user = user.long()
        pos_items = pos_items.long()
        neg_items = torch.tensor(neg_items, dtype=torch.long)
        pos_edge_index = torch.stack([user, pos_items], dim=0)
        pd_mu, pd_logvar, z, v_mu, v_logvar, t_mu, t_logvar, c_mu, c_logvar = self.forward()

        z_v = self.reparametrize(v_mu, v_logvar)
        z_t = self.reparametrize(t_mu, t_logvar)
        z_c = self.reparametrize(c_mu, c_logvar)
        recon_loss = self.recon_loss(z, pos_edge_index, user, neg_items)
        kl_loss = self.kl_loss(pd_mu, pd_logvar)
        loss_multi = recon_loss + kl_loss
        loss_v = self.recon_loss(z_v, pos_edge_index, user, neg_items) + self.kl_loss(v_mu, v_logvar)
        loss_t = self.recon_loss(z_t, pos_edge_index, user, neg_items) + self.kl_loss(t_mu, t_logvar)
        loss_c = self.recon_loss(z_c, pos_edge_index, user, neg_items) + self.kl_loss(c_mu, c_logvar)
        return loss_multi + loss_v + loss_t + loss_c, recon_loss, kl_loss

    def accuracy(self, dataset, topk=10, neg_num=500):
        all_set = set(list(np.arange(neg_num)))
        sum_pre = 0.0
        sum_recall = 0.0
        sum_ndcg = 0.0
        sum_item = 0
        bar = tqdm(total=len(dataset))

        for data in dataset:
            bar.update(1)
            if len(data) < 502:
                continue

            sum_item += 1
            user = data[0]
            neg_items = data[1:501]
            pos_items = data[501:]

            batch_user_tensor = torch.tensor(user).to(self.device)
            batch_pos_tensor = torch.tensor(pos_items).to(self.device)
            batch_neg_tensor = torch.tensor(neg_items).to(self.device)

            batch_neg_tensor = batch_neg_tensor.long()
            batch_user_tensor = batch_user_tensor.long()
            batch_pos_tensor = batch_pos_tensor.long()

            user_embed = self.result_embed[batch_user_tensor]
            pos_v_embed = self.result_embed[batch_pos_tensor]
            neg_v_embed = self.result_embed[batch_neg_tensor]

            num_pos = len(pos_items)
            if num_pos == 0:
                continue
            pos_score = torch.sum(user_embed * pos_v_embed, dim=1)
            neg_score = torch.sum(user_embed * neg_v_embed, dim=1)

            _, index_of_rank_list = torch.topk(torch.cat((neg_score, pos_score)), topk)
            index_set = set([iofr.cpu().item() for iofr in index_of_rank_list])
            num_hit = len(index_set.difference(all_set))
            sum_pre += float(num_hit / topk)
            sum_recall += float(num_hit / num_pos)
            ndcg_score = 0.0
            for i in range(num_pos):
                label_pos = neg_num + i
                if label_pos in index_of_rank_list:
                    index = list(index_of_rank_list.cpu().numpy()).index(label_pos)
                    ndcg_score = ndcg_score + math.log(2) / math.log(index + 2)
            sum_ndcg += ndcg_score / num_pos
        bar.close()

        return sum_pre / sum_item, sum_recall / sum_item, sum_ndcg / sum_item


class ProductOfExperts(torch.nn.Module):
    def __init__(self):
        super(ProductOfExperts, self).__init__()

    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """

    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar, pd_var
