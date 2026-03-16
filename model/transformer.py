from torch import nn
import torch


class MHA(nn.Module):
    def __init__(self, dmodel, nhead):
        super().__init__()
        self.dhead = dmodel // nhead
        self.nhead = nhead
        self.qln = nn.LayerNorm(dmodel)
        self.kvln = nn.LayerNorm(dmodel)
        self.q_proj = nn.Linear(dmodel, dmodel, bias=False)
        self.k_proj = nn.Linear(dmodel, dmodel, bias=False)
        self.v_proj = nn.Linear(dmodel, dmodel, bias=False)
        self.v_gate_proj = nn.Linear(dmodel, dmodel)
        self.out_proj = nn.Linear(dmodel, dmodel, bias=False)

    def forward(self, q, k, v):
        q = self.qln(q)
        q_ = self.q_proj(q)
        k = self.k_proj(self.kvln(k))
        v = self.v_proj(self.kvln(v))

        b, lq, lv = q.shape[0], q.shape[1], v.shape[1]

        q_ = q_.view(b, lq, self.nhead, self.dhead).transpose(1, 2)
        k = k.view(b, lv, self.nhead, self.dhead).transpose(1, 2)
        v = v.view(b, lv, self.nhead, self.dhead).transpose(1, 2)

        attention = q_ @ torch.transpose(k, -2, -1) / self.dhead ** 0.5
        attention = torch.softmax(attention, dim=-1)
        y = attention @ v
        g = torch.sigmoid(self.v_gate_proj(q))
        y = y.transpose(1, 2).contiguous().view(b, lq, -1)
        y = y * g
        y = self.out_proj(y)
        return y


class EncLayer(nn.Module):
    def __init__(self, dmodel, nhead):
        super().__init__()
        self.dmodel = dmodel
        self.mhsa = MHA(dmodel, nhead)
        self.ln = nn.LayerNorm(dmodel)

        self.fc1 = nn.Linear(dmodel, 4 * dmodel)
        self.fc2 = nn.Linear(dmodel * 4, dmodel)

    def forward(self, q, k, v):
        y = v + self.mhsa(q, k, v)
        y_ = torch.nn.functional.gelu(self.fc1(self.ln(y)))
        y = y + self.fc2(y_)
        return y

class DecLayer(nn.Module):
    def __init__(self, dmodel, nhead):
        super().__init__()
        self.dmodel = dmodel
        self.mhsa = MHA(dmodel, nhead)
        self.mhca = MHA(dmodel, nhead)
        self.ln = nn.LayerNorm(dmodel)

        self.fc1 = nn.Linear(dmodel, 4 * dmodel)
        self.fc2 = nn.Linear(dmodel * 4, dmodel)

    def forward(self, q, mem_k, mem_v, skip_sa=False):
        if not skip_sa:
            q = q + self.mhsa(q, q, q)
        q = q + self.mhca(q, mem_k, mem_v)
        y = torch.nn.functional.gelu(self.fc1(self.ln(q)))
        y = q + self.fc2(y)
        return y


if __name__ == '__main__':
    a = torch.randn([4, 128, 256])
    enc = EncLayer(256, 4)
    dec = DecLayer(256, 4)
    y = enc(a, a, a)
    y = dec(y, y, y)
    print(y.shape)