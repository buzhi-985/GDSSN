import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def full_block(in_features, out_features, p_drop=0.0):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )


class MLPEncoder(nn.Module):

    def __init__(self, num_genes=10000, num_hiddens=128, p_drop=0.0):
        super().__init__()
        self.gcn = GCNConv(1, 1, node_dim=0)
        self.encoder = nn.Sequential(
            full_block(num_genes, 1024, p_drop),
            full_block(1024, num_hiddens, p_drop),
        )

    def forward(self, data, edge_index):

        for i in range(data.shape[0]):
            x = torch.as_tensor(data[i].reshape(data.shape[1], 1),dtype=torch.float)
            y = self.gcn(x, edge_index)
            y = F.relu(y).t()
            if i == 0:
                gcn_out = y
            else:
                gcn_out = torch.cat([gcn_out, y], dim=0)
        x = self.encoder(gcn_out)
        return x



class MoCo(nn.Module):

    def __init__(self, e, base_encoder, num_genes=10000, dim=16, r=512, m=0.999, T=0.2):
        super(MoCo, self).__init__()

        self.r = r
        self.m = m
        self.T = T
        self.edgeindex = e

        print("num_genes={}, num_hiddens={}".format(num_genes, dim))
        self.encoder_q = base_encoder(num_genes=num_genes, num_hiddens=dim)
        self.encoder_k = base_encoder(num_genes=num_genes, num_hiddens=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False


        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.r % batch_size == 0


        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.r

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k=None, is_eval=False, index=None):
        if is_eval:
            k = self.encoder_k(im_q, self.edgeindex)
            k = nn.functional.normalize(k, dim=1)
            return k

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k, self.edgeindex)
            k = nn.functional.normalize(k, dim=1)

        q = self.encoder_q(im_q, self.edgeindex)
        q = nn.functional.normalize(q, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        self._dequeue_and_enqueue(k)

        return logits, labels, None, None
