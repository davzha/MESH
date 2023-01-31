import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from mesh import sinkhorn, minimize_entropy_of_sinkhorn


@torch.jit.script
def l2_distance(x, y):
    return torch.cdist(x, y, p=2.0)


@torch.jit.script
def cosine_distance(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 1.0 - torch.bmm(x, y.transpose(1, 2))


@torch.jit.script
def dot_prod(x, y):
    factor = x.size(-1) ** -0.5
    return factor * torch.bmm(x, y.transpose(1, 2))


pairwise_distances = {
    "l2": l2_distance,
    "cosine": cosine_distance,
    "dot_prod": dot_prod,
}


class SlotAttentionVariant(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.num_iterations = cfg.n_iters
        # self.cfg.n_slots = cfg.n_slots
        # self.cfg.d_slot = cfg.d_slot  # number of hidden layers in slot dimensions
        # self.lambda_sh = cfg.lambda_sh

        self.pairwise_distance = pairwise_distances[cfg.pairwise_distance]

        self.norm_inputs = nn.LayerNorm(cfg.d_in)
        self.norm_slots = nn.LayerNorm(cfg.d_slot)
        self.norm_mlp = nn.LayerNorm(cfg.d_slot)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(cfg.d_slot, cfg.d_slot, bias=False)
        self.project_k = nn.Linear(cfg.d_in, cfg.d_slot, bias=False)
        self.project_v = nn.Linear(cfg.d_in, cfg.d_slot, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(cfg.d_slot, cfg.d_slot)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_slot, cfg.d_mlp),
            nn.ReLU(),
            nn.Linear(cfg.d_mlp, cfg.d_slot),
        )

        if cfg.attention_type.lower() != "original_slot_attention":
            self.mlp_slot_marginals = nn.Sequential(
                nn.Linear(cfg.d_slot, cfg.d_mlp),
                nn.ReLU(),
                nn.Linear(cfg.d_mlp, 1, bias=False),
            )
            self.mlp_input_marginals = nn.Sequential(
                nn.Linear(cfg.d_in, cfg.d_mlp),
                nn.ReLU(),
                nn.Linear(cfg.d_mlp, 1, bias=False),
            )
            if cfg.attention_type.lower() == "mesh":
                self.lr = (
                    nn.Parameter(torch.ones(1)) if cfg.learn_mesh_lr else cfg.mesh_lr
                )

        if cfg.init_slot_method == "learned_random":
            self.slots_mu = nn.Parameter(torch.zeros((1, 1, cfg.d_slot)))
            nn.init.xavier_uniform_(
                self.slots_mu, gain=nn.init.calculate_gain("linear")
            )
            self.slots_log_sigma = nn.Parameter(torch.zeros((1, 1, cfg.d_slot)))
            nn.init.xavier_uniform_(
                self.slots_log_sigma, gain=nn.init.calculate_gain("linear")
            )
        elif cfg.init_slot_method == "fixed_random":
            self.register_buffer(
                "slots_mu",
                nn.init.xavier_uniform_(
                    torch.zeros((1, 1, cfg.d_slot)),
                    gain=nn.init.calculate_gain("linear"),
                ),
            )
            self.register_buffer(
                "slots_log_sigma",
                nn.init.xavier_uniform_(
                    torch.zeros((1, 1, cfg.d_slot)),
                    gain=nn.init.calculate_gain("linear"),
                ),
            )
        else:
            raise ValueError(f"Unknown init_slots: {cfg.init_slots}.")

    def forward(self, inputs: Tensor):
        # `inputs` has shape [batch_size, num_inputs, d_input].
        batch_size, num_inputs, d_input = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        # Initialize the slots. Shape: [batch_size, num_slots, d_slots].
        slots_init = torch.randn(
            (batch_size, self.cfg.n_slots, self.cfg.d_slot),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init

        attention_type = self.cfg.attention_type.lower()
        if attention_type != "original_slot_attention":
            scale_marginals = self.cfg.scale_marginals
            a = scale_marginals * self.mlp_input_marginals(inputs).squeeze(2).softmax(
                dim=1
            )

            if attention_type == "mesh":
                noise = torch.randn(
                    batch_size, num_inputs, self.cfg.n_slots, device=slots.device
                )
            sh_u = sh_v = None

        for i in range(self.cfg.n_sa_iters):
            if self.cfg.detach_slots:
                slots = slots.detach()
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots)

            cost = self.pairwise_distance(k, q)

            if attention_type != "original_slot_attention":
                b = scale_marginals * self.mlp_slot_marginals(slots).squeeze(2).softmax(
                    dim=1
                )

            if attention_type == "mesh":
                cost, sh_u, sh_v = minimize_entropy_of_sinkhorn(
                    cost, a, b, noise=noise, mesh_lr=self.lr, **self.cfg.mesh_args
                )
                if not self.cfg.mesh_args.reuse_u_v:
                    sh_v = sh_u = None
                attn, *_ = sinkhorn(
                    cost,
                    a,
                    b,
                    u=sh_u,
                    v=sh_v,
                    n_sh_iters=self.cfg.n_sh_iters,
                    temperature=self.cfg.temperature,
                )

            elif attention_type == "sinkhorn":
                attn = sinkhorn(
                    cost,
                    a,
                    b,
                    n_sh_iters=self.cfg.n_sh_iters,
                    temperature=self.cfg.temperature,
                )[0]

            elif attention_type == "original_slot_attention":
                attn = F.softmax(cost, dim=-1)
                attn = attn + 1e-8
                attn = attn / torch.sum(attn, dim=1, keepdim=True)

            else:
                raise ValueError(f"unknown attention type {self.cfg.ot_mode}")
            updates = torch.matmul(attn.transpose(1, 2), v)

            slots = self.gru(
                updates.view(batch_size * self.cfg.n_slots, self.cfg.d_slot),
                slots_prev.view(batch_size * self.cfg.n_slots, self.cfg.d_slot),
            )
            slots = slots.view(batch_size, self.cfg.n_slots, self.cfg.d_slot)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


def build_grid(resolution):
    grid = torch.stack(
        torch.meshgrid(
            *[torch.linspace(0.0, 1.0, steps=r) for r in resolution], indexing="ij"
        ),
        dim=-1,
    )
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


class SoftPositionEmbed(nn.Module):
    def __init__(self, dim, res):
        super().__init__()
        self.dense = nn.Linear(len(res) * 2, dim)
        self.register_buffer("grid", build_grid(res))

    def forward(self, inputs):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        return inputs + emb_proj


class CNNEncoder(nn.Sequential):
    def __init__(self, d_hid, d_out, resolution):
        super().__init__(
            nn.Conv2d(3, d_hid, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(d_hid, d_hid, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(d_hid, d_hid, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(d_hid, d_hid, 5, padding=2),
            nn.ReLU(),
            SoftPositionEmbed(d_hid, resolution),
            Rearrange("b c h w -> b (h w) c"),
            nn.LayerNorm(d_hid),
            nn.Linear(d_hid, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_out),
        )


class SpatialBroadcast(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution

    def forward(self, x):
        return x.view(-1, x.size(-1), 1, 1).expand(-1, -1, *self.resolution)


class CNNDecoder(nn.Sequential):
    def __init__(self, d_slot, d_hid, broadcast_resolution):
        super().__init__(
            SpatialBroadcast(broadcast_resolution),
            SoftPositionEmbed(d_slot, broadcast_resolution),
            nn.ConvTranspose2d(d_slot, d_hid, 5, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(d_hid, d_hid, 5, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(d_hid, d_hid, 5, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(d_hid, 4, 3, padding=1),
        )


class SlotAttentionObjectDiscovery(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.encoder = CNNEncoder(
            cfg.encoder.d_hid,
            cfg.encoder.d_out,
            cfg.encoder.resolution,
        )
        self.decoder = CNNDecoder(
            cfg.decoder.d_in,
            cfg.decoder.d_hid,
            cfg.decoder.broadcast_resolution,
        )
        self.slot_attention = SlotAttentionVariant(cfg.slot_attention)

    def forward(self, x):
        bsz, c, h, w = x.shape
        x = self.encoder(x)
        slots = self.slot_attention(x)
        x = self.decoder(slots)
        x = rearrange(x, "(b n) c h w -> b n c h w", b=bsz)
        rgb_channels, alpha_masks = x.split((c, 1), dim=2)
        alpha_masks = alpha_masks.softmax(dim=1)
        reconstruction = (rgb_channels * alpha_masks).sum(dim=1)

        return reconstruction, rgb_channels, alpha_masks, slots
