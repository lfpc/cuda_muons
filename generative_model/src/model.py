import torch
import torch.nn as nn
import pyro.distributions as dist
import pyro.distributions.transforms as T


class ContextEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, context):
        return self.net(context)


class ConditionalFlowModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        ctx_out = config["context_out_dim"]
        self.embedder = ContextEmbedder(
            in_dim=2,
            hidden_dims=config["context_hidden_dims"],
            out_dim=ctx_out,
        )
        self.transforms = nn.ModuleList([
            T.conditional_spline_autoregressive(
                input_dim=2,
                context_dim=ctx_out,
                hidden_dims=config["transform_hidden_dims"],
                count_bins=config["count_bins"],
                bound=config["spline_bound"],
                order="linear",
            )
            for _ in range(config["n_transforms"])
        ])
        self._perm = torch.tensor([1, 0])

    def _conditioned_transform_list(self, ctx_emb):
        transforms = []
        for i, t in enumerate(self.transforms):
            transforms.append(t.condition(ctx_emb))
            if i < len(self.transforms) - 1:
                transforms.append(T.Permute(self._perm.to(ctx_emb.device)))
        return transforms

    def log_prob(self, x, context):
        ctx_emb = self.embedder(context)
        transform_list = self._conditioned_transform_list(ctx_emb)
        base = dist.Normal(
            torch.zeros(2, device=ctx_emb.device),
            torch.ones(2, device=ctx_emb.device),
        ).to_event(1)
        flow = dist.TransformedDistribution(base, transform_list)
        return flow.log_prob(x)

    @torch.no_grad()
    def sample(self, context):
        ctx_emb = self.embedder(context)
        z = torch.randn(ctx_emb.shape[0], 2, device=ctx_emb.device)
        perm = self._perm.to(ctx_emb.device)
        for i, t in enumerate(self.transforms):
            z = t.condition(ctx_emb)(z)
            if i < len(self.transforms) - 1:
                z = z[..., perm]
        return z

    def clear_cache(self):
        for t in self.transforms:
            if hasattr(t, "_cache"):
                t._cache.clear()
