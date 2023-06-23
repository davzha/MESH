import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce


def pairwise_IoU(pred_mask, gt_mask):
    pred_mask = repeat(pred_mask, "... n c -> ... 1 n c")
    gt_mask = repeat(gt_mask, "... n c -> ... n 1 c")
    intersection = pred_mask & gt_mask
    union = pred_mask | gt_mask
    iou = intersection.sum(-1) / union.sum(-1)
    return iou


def compute_IoU(pred_mask, gt_mask):
    if pred_mask.ndim == 4:
        time = pred_mask.size(1)
        pred_mask = rearrange(pred_mask, "b t n c -> (b t) n c")
        gt_mask = rearrange(gt_mask, "b t n c -> (b t) n c")
    else:
        time = None
        
    # assumes shape: batch_size, set_size, channels
    is_padding = (gt_mask == 0).all(-1)


    # discretized 2d mask for hungarian matching
    pred_mask_id = torch.argmax(pred_mask, -2)
    pred_mask_disc = rearrange(
        F.one_hot(pred_mask_id, pred_mask.size(-2)).to(torch.float32), "... c n -> ... n c"
    )

    # treat as if no padding in gt_mask
    pIoU = pairwise_IoU(pred_mask_disc.bool(), gt_mask.bool())
    pIoU_inv = 1 - pIoU
    pIoU_inv[is_padding] = 1e3
    pIoU_inv_ = pIoU_inv.detach().cpu().numpy()

    # hungarian matching
    indices = np.array([linear_sum_assignment(p) for p in pIoU_inv_])
    indices_ = pred_mask.size(-2) * indices[:, 0] + indices[:, 1]
    indices_ = torch.from_numpy(indices_).to(device=pred_mask.device)
    IoU = torch.gather(rearrange(pIoU, "... n m -> ... (n m)"), 1, indices_)
    IoU[is_padding] = 0
    mIoU = (IoU * ~is_padding).sum(-1) / (~is_padding).sum(-1)
    if time is not None:
        mIoU = rearrange(mIoU, "(b t) -> b t", t=time)
    return mIoU


# https://github.dev/vadimkantorov/yet_another_pytorch_slot_attention
@torch.no_grad()
def adjusted_rand_index(true_mask, pred_mask):
    *_, n_points, n_true_groups = true_mask.shape
    n_pred_groups = pred_mask.shape[-1]
    assert not (
        n_points <= n_true_groups and n_points <= n_pred_groups
    ), "adjusted_rand_index requires n_groups < n_points. We don't handle the special cases that can occur when you have one cluster per datapoint."

    true_group_ids = torch.argmax(true_mask, -1)
    pred_group_ids = torch.argmax(pred_mask, -1)
    true_mask_oh = true_mask.to(torch.float32)
    pred_mask_oh = F.one_hot(pred_group_ids, n_pred_groups).to(torch.float32)

    n_points = torch.sum(true_mask_oh, dim=[-2, -1]).to(torch.float32)

    nij = torch.einsum("...ji,...jk->...ki", pred_mask_oh, true_mask_oh)
    a = torch.sum(nij, dim=-2)
    b = torch.sum(nij, dim=-1)

    rindex = torch.sum(nij * (nij - 1), dim=[-2, -1])
    aindex = torch.sum(a * (a - 1), dim=-1)
    bindex = torch.sum(b * (b - 1), dim=-1)
    expected_rindex = aindex * bindex / (n_points * (n_points - 1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)
    _mask_bg = true_mask.sum(dim=-1) == 0
    both_single_cluster = torch.logical_and(
        all_equal(true_group_ids, _mask_bg), all_equal(pred_group_ids, _mask_bg)
    )

    return torch.where(both_single_cluster, torch.ones_like(ari), ari)


def all_equal(values, mask):
    """mask[...] == True iff it should be ignored"""
    if values.ndim == 3:
        time = values.size(1)
        values = rearrange(values, "b t n -> (b t) n")
        mask = rearrange(mask, "b t n -> (b t) n")
    else:
        time = None
    ref_val = torch.gather(values, 1, mask.float().argmin(dim=-1).unsqueeze(1))
    eq = values == ref_val
    eq = torch.logical_or(eq, mask)
    eq = torch.all(eq, dim=-1)
    if time is not None:
        eq = rearrange(eq, "(b t) -> b t", t=time)
    return eq


def time_consistency_accuracy(pred_mask, gt_mask, attributes):
    # pred_mask: b t m hw
    # gt_mask: b t n hw
    is_padding = (gt_mask == 0).all(-1)

    # discretized 2d mask for hungarian matching
    pred_mask_id = torch.argmax(pred_mask, -2)
    pred_mask_disc = rearrange(
        F.one_hot(pred_mask_id, pred_mask.size(2)).to(torch.float32),
        "... c n -> ... n c")

    # treat as if no padding in gt_mask
    pIoU = pairwise_IoU(pred_mask_disc.bool(), gt_mask.bool())
    pIoU_inv = 1 - pIoU
    pIoU_inv[is_padding] = 1e3
    pIoU_inv_ = rearrange(pIoU_inv, "b t n m -> (b t) n m").detach().cpu().numpy()
    indices = np.array([linear_sum_assignment(p) for p in pIoU_inv_])
    indices = torch.from_numpy(indices).to(device=pred_mask.device)
    slot_id = rearrange(indices[:, -1], "(b t) n -> b t n", b=pred_mask.shape[0])

    # check which objects are the same in the two frames modulo padding
    same_obj = (rearrange(attributes[:,0], "b n c -> b n 1 c") == rearrange(attributes[:,1], "b n c -> b 1 n c")).all(-1)
    is_padding = rearrange(is_padding[:,0], "b n -> b n 1") | rearrange(is_padding[:,1], "b n -> b 1 n")
    same_obj = same_obj & ~is_padding

    # compute accuracy and catch the cases where there are no overlaps between the two frames
    true_pos = same_obj & (rearrange(slot_id[:,0], "b n -> b n 1") == rearrange(slot_id[:,1], "b n -> b 1 n"))
    n_overlaps = reduce(same_obj, "b n m -> b", "sum")
    acc = reduce(true_pos, "b n m -> b", "sum") / n_overlaps
    return acc


def hungarian_loss(pred, target, loss_fn=F.smooth_l1_loss):
    pdist = loss_fn(
        pred.unsqueeze(1).expand(-1, target.size(1), -1, -1), 
        target.unsqueeze(2).expand(-1, -1, pred.size(1), -1),
        reduction='none').mean(3)

    pdist_ = pdist.detach().cpu().numpy()

    indices = np.array([linear_sum_assignment(p) for p in pdist_])

    indices_ = indices.shape[2] * indices[:, 0] + indices[:, 1]
    losses = torch.gather(pdist.flatten(1,2), 1, torch.from_numpy(indices_).to(device=pdist.device))
    total_loss = losses.mean(1)

    return total_loss, dict(indices=indices)