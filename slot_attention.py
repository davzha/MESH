import math
import torch
from torch import nn
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
    def __init__(self, d_in, d_slot, n_slots, d_mlp, attention_type, n_sa_iters, pairwise_distance, temperature,
                scale_marginals, n_sh_iters, learn_mesh_lr, mesh_lr, init_slot_method, detach_slots, mesh_args):
        super().__init__()
        self.d_slot = d_slot
        self.n_slots = n_slots
        self.attention_type = attention_type
        self.scale_marginals = scale_marginals
        self.n_sa_iters = n_sa_iters
        self.n_sh_iters = n_sh_iters
        self.detach_slots = detach_slots
        self.mesh_args = mesh_args
        self.temperature = temperature
        self.pairwise_distance = pairwise_distances[pairwise_distance]

        self.norm_inputs = nn.LayerNorm(d_in)
        self.norm_slots = nn.LayerNorm(d_slot)
        self.norm_mlp = nn.LayerNorm(d_slot)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(d_slot, d_slot, bias=False)
        self.project_k = nn.Linear(d_in, d_slot, bias=False)
        self.project_v = nn.Linear(d_in, d_slot, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(d_slot, d_slot)
        self.mlp = nn.Sequential(
            nn.Linear(d_slot, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_slot),
        )

        if attention_type.lower() != "original_slot_attention":
            self.mlp_slot_marginals = nn.Sequential(
                nn.Linear(d_slot, d_mlp),
                nn.ReLU(),
                nn.Linear(d_mlp, 1, bias=False),
            )
            self.mlp_input_marginals = nn.Sequential(
                nn.Linear(d_in, d_mlp),
                nn.ReLU(),
                nn.Linear(d_mlp, 1, bias=False),
            )
            if attention_type.lower() == "mesh":
                self.lr = (
                    nn.Parameter(torch.ones(1)) if learn_mesh_lr else mesh_lr
                )

        if init_slot_method == "learned_random":
            self.slots_mu = nn.Parameter(torch.zeros((1, 1, d_slot)))
            nn.init.xavier_uniform_(
                self.slots_mu, gain=nn.init.calculate_gain("linear")
            )
            self.slots_log_sigma = nn.Parameter(torch.zeros((1, 1, d_slot)))
            nn.init.xavier_uniform_(
                self.slots_log_sigma, gain=nn.init.calculate_gain("linear")
            )
        elif init_slot_method == "fixed_random":
            self.register_buffer(
                "slots_mu",
                nn.init.xavier_uniform_(
                    torch.zeros((1, 1, d_slot)),
                    gain=nn.init.calculate_gain("linear"),
                ),
            )
            self.register_buffer(
                "slots_log_sigma",
                nn.init.xavier_uniform_(
                    torch.zeros((1, 1, d_slot)),
                    gain=nn.init.calculate_gain("linear"),
                ),
            )
        else:
            raise ValueError(f"Unknown init_slots: {init_slot_method}.")

    def forward(self, inputs, initial_slots = None):
        # `inputs` has shape [batch_size, num_inputs, d_input].
        batch_size, num_inputs, d_input = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        # Initialize the slots. Shape: [batch_size, num_slots, d_slots].
        if initial_slots is None:
            slots_init = torch.randn(
                (batch_size, self.n_slots, self.d_slot),
                device=inputs.device,
                dtype=inputs.dtype,
            )
            slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init
        else:
            slots = initial_slots

        attention_type = self.attention_type.lower()
        if attention_type != "original_slot_attention":
            scale_marginals = self.scale_marginals
            a = scale_marginals * self.mlp_input_marginals(inputs).squeeze(2).softmax(
                dim=1
            )

            if attention_type == "mesh":
                noise = torch.randn(
                    batch_size, num_inputs, self.n_slots, device=slots.device
                )
            sh_u = sh_v = None

        for i in range(self.n_sa_iters):
            if self.detach_slots:
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
                    cost, a, b, noise=noise, mesh_lr=self.lr, **self.mesh_args
                )
                if not self.mesh_args.reuse_u_v:
                    sh_v = sh_u = None
                attn, *_ = sinkhorn(
                    cost,
                    a,
                    b,
                    u=sh_u,
                    v=sh_v,
                    n_sh_iters=self.n_sh_iters,
                    temperature=self.temperature,
                )

            elif attention_type == "sinkhorn":
                attn = sinkhorn(
                    cost,
                    a,
                    b,
                    n_sh_iters=self.n_sh_iters,
                    temperature=self.temperature,
                )[0]

            elif attention_type == "original_slot_attention":
                attn = F.softmax(cost, dim=-1)
                attn = attn + 1e-8
                attn = attn / torch.sum(attn, dim=1, keepdim=True)

            else:
                raise ValueError(f"unknown attention type {attention_type}")
            updates = torch.matmul(attn.transpose(1, 2), v)

            slots = self.gru(
                updates.view(batch_size * self.n_slots, self.d_slot),
                slots_prev.view(batch_size * self.n_slots, self.d_slot),
            )
            slots = slots.view(batch_size, self.n_slots, self.d_slot)
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
    def __init__(self, encoder, decoder, slot_attention):
        super().__init__()
        self.encoder = CNNEncoder(**encoder)
        self.decoder = CNNDecoder(**decoder)
        self.slot_attention = SlotAttentionVariant(**slot_attention)

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
    

class TransitionWrapper(nn.Module):
    def __init__(self, slot_model, d_slot, noise_sigma=0., learned_noise=False, learned_noise_var=0.1):
        super().__init__()
        assert noise_sigma == 0. or ~learned_noise, "can't have both learned and fixed noise"
        self.slot_model = slot_model
        self.noise_sigma = noise_sigma
        self.learned_noise = learned_noise
        self.learned_noise_var = learned_noise_var

        if learned_noise:
            self.noise_mlp = nn.Sequential(
                nn.LayerNorm(d_slot),
                nn.Linear(d_slot, d_slot),
                nn.ReLU(),
                nn.Linear(d_slot, 2*d_slot),
            )
        self.d_slot = d_slot

    def forward(self, x):
        # x has shape (batch, time, set, ...)
        timesteps = x.size(1)
        all_slots = []
        slots = None
        reg_loss = 0.

        for i in range(timesteps):
            if slots is not None:
                # optionally add noise to separate slots
                if self.noise_sigma > 0.:
                    slots = slots + self.noise_sigma * torch.randn_like(slots)
                elif self.learned_noise:
                    normal_param = self.noise_mlp(slots)
                    mu = normal_param[..., :self.d_slot]
                    log_var = normal_param[..., self.d_slot:]
                    slots = mu + torch.randn_like(slots) * torch.exp(log_var)
                    loss_kl = math.log(math.sqrt(self.learned_noise_var)) - 0.5*log_var + 0.5*log_var.exp()/self.learned_noise_var - 0.5
                    reg_loss = reg_loss + loss_kl.mean()

            slots = self.slot_model(x[:, i], initial_slots=slots)
            all_slots.append(slots)


        return torch.stack(all_slots, dim=1), reg_loss
    

class SlotAttentionObjectDiscoveryVideo(nn.Module):
    def __init__(self, encoder, decoder, slot_attention, transition_wrapper):
        super().__init__()
        self.encoder = CNNEncoder(**encoder)
        self.decoder = CNNDecoder(**decoder)
        self.slot_attention = TransitionWrapper(SlotAttentionVariant(**slot_attention), **transition_wrapper)

    def forward(self, x):
        bsz, time, c, h, w = x.shape

        x = x.flatten(0, 1)
        x = self.encoder(x)
        x = x.view(bsz, time, *x.size()[1:])

        slots, reg_loss = self.slot_attention(x)
        # n_slots = slots.size(-1)
        # slots = repeat(slots, "b t n c -> (b t n) c w h", w=w, h=h).contiguous()
        # x = slots.flatten(0,1)[:,:,None,None].expand(bsz*time*n_slots, slots.size(-1), w, h)
        slots = slots.flatten(0,1)
        x = self.decoder(slots)
        x = rearrange(x, "(b t n) c h w -> b t n c h w", b=bsz, t=time)
        rgb_channels, alpha_masks = x.split((c, 1), dim=-3)
        alpha_masks = alpha_masks.softmax(dim=2)
        reconstruction = (rgb_channels * alpha_masks).sum(dim=2)

        return reconstruction, rgb_channels, alpha_masks, slots, reg_loss
