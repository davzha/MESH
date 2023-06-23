#%%
from pathlib import Path
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import hydra
import wandb

import data
import losses
import slot_attention
from lr_scheduler import SA_LRScheduler


class ObjectDiscoveryModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.net = self.get_model()
        self.trainset = hydra.utils.instantiate(cfg.data.train)
        self.valset = hydra.utils.instantiate(cfg.data.val)
        self.testset = hydra.utils.instantiate(cfg.data.test)

    def get_model(self):
        net = hydra.utils.instantiate(self.hparams.model)
        if self.hparams.compile:
            net = torch.compile(net)
        return net

    def forward(self, x):
        if len(x) == 3:
            input, gt_output, gt_masks = x
            attributes = None
            reg_loss = 0.0
            output, _, masks, *_ = self.net(input)
        else:
            input, gt_output, gt_masks, attributes = x
            output, _, masks, *_, reg_loss = self.net(input)

        return output, gt_output, masks, gt_masks, attributes, reg_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "/train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "/val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "/test")

    def step(self, batch, batch_idx, suffix):
        output, gt_output, masks, gt_masks, attributes, reg_loss = self(batch)

        # if gt_output.ndim == 5:
        #     n_frames = gt_output.size(1)
        #     gt_output = gt_output.flatten(0, 1)
        #     gt_masks = gt_masks.flatten(0, 1)

        loss = F.mse_loss(output, gt_output)
        loss = loss + reg_loss

        ari = losses.adjusted_rand_index(
            rearrange(gt_masks, "... n 1 h w -> ... (h w) n"),
            rearrange(masks, "... n 1 h w -> ... (h w) n"),
        ).mean()

        log_dict = dict(loss=loss, ari=ari)

        if not self.training:
            miou = losses.compute_IoU(
                pred_mask=rearrange(masks, "... 1 h w -> ... (h w)"),
                gt_mask=rearrange(gt_masks, "... 1 h w-> ... (h w)"),
            )
            log_dict["miou"] = miou.mean()

            if len(batch) > 3:
                tca = losses.time_consistency_accuracy(
                    pred_mask=rearrange(masks, "... n 1 h w -> ... n (h w)"),
                    gt_mask=rearrange(gt_masks, "... n 1 h w -> ... n (h w)"),
                    attributes=attributes
                )
                log_dict["tca"] = tca.mean()

        if batch_idx % self.hparams.log_img_freq == 0:
            self.plot_progress(
                output,
                masks,
                gt_output,
                suffix=suffix,
            )

        self.log_dict(
            {k + suffix: v for k, v in log_dict.items()},
            on_step=suffix == "/train",
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        parameters = self.net.parameters()
        opt = torch.optim.Adam(parameters, lr=self.hparams.opt.lr)
        optimizers = {
            "optimizer": opt,
        }

        if self.hparams.opt.use_lr_scheduler:
            optimizers["lr_scheduler"] = {
                "interval": "step",
                "frequency": 1,
                "scheduler": SA_LRScheduler(opt),
            }

        return optimizers

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.data.num_data_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.data.num_data_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.data.num_data_workers,
        )

    def plot_progress(self, pred, masks, gt, n_examples=4, suffix=""):
        if pred.ndim == 5:
            # n_frames = pred.size(1)
            pred = pred.flatten(0, 1)
            masks = masks.flatten(0, 1)
            gt = gt.flatten(0, 1)
        pred = pred.cpu().detach().permute(0, 2, 3, 1) + 1
        pred = (pred / 2).clamp(0, 1)
        masks = masks.cpu().detach()
        gt = gt.cpu().detach().permute(0, 2, 3, 1) + 1
        gt = (gt / 2).clamp(0, 1)
        fig, axs = plt.subplots(n_examples, 2 + masks.size(1), squeeze=False)
        fig.subplots_adjust(hspace=-0.5, wspace=0.1)
        for p, ms, g, ax_pair in zip(pred, masks, gt, axs):
            ax1, *ax_middle, ax2 = ax_pair
            ax1.imshow(p, vmin=0, vmax=1, interpolation="nearest")
            ax1.axis("off")
            for m, ax in zip(ms, ax_middle):
                ax.imshow(m[0], vmin=0, vmax=1, cmap="gray", interpolation="nearest")
                ax.axis("off")
            ax2.imshow(g, vmin=0, vmax=1, interpolation="nearest")
            ax2.axis("off")
        plt.axis("off")

        self.logger.experiment.log(
            {
                f"reconstruction{suffix}": wandb.Image(fig),
                "epoch": self.current_epoch,
                "global_step": self.global_step,
            },
            commit=False,
        )

        plt.close(fig)


def train(cfg):
    model = ObjectDiscoveryModel(cfg)

    wandb.init(
        name=cfg.name,
        project=cfg.project,
        reinit=False,
        config=cfg,
        entity=cfg.wandb_entity,
        # settings=wandb.Settings(start_method="fork"),
    )
    logger = WandbLogger(log_model=True)
    logger.watch(model.net)
    wandb.config.update(cfg)

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[
            ModelCheckpoint(monitor="loss/val", save_last=True),
            LearningRateMonitor(),
        ],
    )
    trainer.fit(
        model, ckpt_path=Path(cfg.ckpt_path).resolve() if cfg.ckpt_path else None
    )
    return trainer


def test(cfg, trainer=None):
    ckpt_path = None
    if trainer is None:
        trainer = pl.Trainer(gpus=cfg.opt.n_gpus, num_nodes=1, inference_mode=False)
        ckpt_path = Path(cfg.ckpt_path).resolve()
    trainer.test(ckpt_path=ckpt_path)


@hydra.main(config_path="config", config_name="object_discovery.yaml", version_base=None)
def main(cfg):
    pl.seed_everything(cfg.seed)
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    if cfg.tf32:
        torch.set_float32_matmul_precision("high")

    trainer = None
    if not cfg.eval_only:
        trainer = train(cfg)

    test(cfg, trainer=trainer)


#%%
if __name__ == "__main__":
    OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
    main()
