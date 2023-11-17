from pathlib import Path
import torch
from models.yolo import ModelEMA

# Assuming you have defined the pruner class (Pruner) and set up other configurations (device, opt, etc.)

yolo_pruner = Pruner(model, device, opt)

for idx, epoch in enumerate(range(start_epoch, epochs)):
    if (idx + 1) % opt.num_epochs_to_prune == 0:
        yolo_pruner.step(model, device)
        
        # Save the pruned model weights after each pruning iteration
        torch.save(model.state_dict(), 'path/to/save/pruned_weights_iter_{}.pt'.format(idx + 1))
        
        ema = ModelEMA(model) if rank in [-1, 0] else None

    model.train()

    # Rest of the code for training and saving checkpoints remains unchanged
    # ...

    torch.save(ckpt, last)
    if best_fitness == fi:
        torch.save(ckpt, best)
    if (best_fitness == fi) and (epoch >= 200):
        torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
    if epoch == 0:
        torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
    elif ((epoch + 1) % 25) == 0:
        torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
    elif epoch >= (epochs - 5):
        torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
    if wandb_logger.wandb:
        if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
            wandb_logger.log_model(
                last.parent, opt, epoch, fi, best_model=best_fitness == fi)
    del ckpt






torch.save(model.state_dict(), 'path/to/save/pruned_weights_iter_{}.pt'.format(idx + 1))
save_path = 'run/weights/pruned_weights_iter_{}.pt'.format(idx + 1)
torch.save(model.state_dict(), save_path)

