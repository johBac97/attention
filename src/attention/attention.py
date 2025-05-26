import torch


class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self._embed_dim = embed_dim

        self._q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self._k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self._v_proj = torch.nn.Linear(embed_dim, embed_dim)

        self._o_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.tensor):
        Q = self._q_proj(x)
        K = self._k_proj(x)
        V = self._v_proj(x)

        scores = torch.bmm(Q, K.transpose(1, 2)) / (self._embed_dim**0.5)

        attention_weights = scores.softmax(dim=-1)

        attention_output = torch.bmm(attention_weights, V)

        # Project into output
        output = self._o_proj(attention_output)

        return output, attention_weights
