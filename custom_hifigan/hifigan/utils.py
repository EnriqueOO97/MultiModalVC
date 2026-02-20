import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt


def get_padding(k, d):
    return int((k * d - d) / 2)


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def save_checkpoint(
    checkpoint_dir,
    generator,
    discriminator,
    optimizer_generator,
    optimizer_discriminator,
    scheduler_generator,
    scheduler_discriminator,
    step,
    loss,
    best,
    logger,
):
    is_ddp = isinstance(generator, torch.nn.parallel.DistributedDataParallel)
    state = {
        "generator": {
            "model": generator.module.state_dict() if is_ddp else generator.state_dict(),
            "optimizer": optimizer_generator.state_dict(),
            "scheduler": scheduler_generator.state_dict(),
        },
        "discriminator": {
            "model": discriminator.state_dict(),
            "optimizer": optimizer_discriminator.state_dict(),
            "scheduler": scheduler_discriminator.state_dict(),
        },
        "step": step,
        "loss": loss,
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / f"model-{step}.pt"
    torch.save(state, checkpoint_path)
    if best:
        best_path = checkpoint_dir / "model-best.pt"
        torch.save(state, best_path)
    logger.info(f"Saved checkpoint: {checkpoint_path.stem}")


def load_checkpoint(
    load_path,
    generator,
    discriminator,
    optimizer_generator,
    optimizer_discriminator,
    scheduler_generator,
    scheduler_discriminator,
    rank,
    logger,
    finetune=False,
):
    logger.info(f"Loading checkpoint from {load_path}")
    checkpoint = torch.load(load_path, map_location={"cuda:0": f"cuda:{rank}"})
    incompatible_keys = generator.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint["generator"]["model"].items()}, strict=False)
    print('incompatible keys:\n')
    print(incompatible_keys)
    discriminator.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint["discriminator"]["model"].items()})
    if not finetune:
        try:
            optimizer_generator.load_state_dict(checkpoint["generator"]["optimizer"])
        except:
            print('direct loading failed. Giving up param_groups.')
            # state only
            optimizer_generator.load_state_dict({
                'state': checkpoint["generator"]["optimizer"]['state'],
                'param_groups': optimizer_generator.state_dict()['param_groups']
            })
        scheduler_generator.load_state_dict(checkpoint["generator"]["scheduler"])
        optimizer_discriminator.load_state_dict(
            checkpoint["discriminator"]["optimizer"]
        )
        scheduler_discriminator.load_state_dict(
            checkpoint["discriminator"]["scheduler"]
        )
    return checkpoint["step"], checkpoint["loss"]
