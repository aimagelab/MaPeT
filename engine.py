import datetime
import logging
import os
import time
import torch
import torchvision.utils
from collections import OrderedDict
from contextlib import suppress
from timm import utils
from timm.models import model_parameters

from data.clipped_imagenet import post_processing_images

_logger = logging.getLogger('engine')

# --------------------------
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
DEBUG_BATCHES = 10


# --------------------------

def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None, **kwargs
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    MB = 1024.0 * 1024.0

    model.train()
    optimizer.zero_grad()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader) // args.iters_to_accumulate

    additional_losses = {}

    for batch_idx, data in enumerate(loader):

        if DEBUG and batch_idx == DEBUG_BATCHES:
            break

        if "mapet" in args.model:
            input, _, target, mask_query, mask_content, index_vector, upper_predictions = data
            target_selection = index_vector * upper_predictions
            target = target[target_selection[:, 1:]].reshape(input.shape[0], -1).long()
        else:
            input, target = data

        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if "mapet" in args.model:
                mask_query, mask_content = mask_query.cuda(), mask_content.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            # --- forward to extract model output(s) ---
            if "mapet" in args.model:
                output = model(input, mask_query, mask_content, index_vector)
                output = output.permute(1, 0, 2)[:,
                         upper_predictions[index_vector].reshape(input.shape[0], -1)].reshape(8192, input.shape[0],
                                                                                              -1).permute(1, 0, 2)
            else:
                # torch.distributed.barrier()
                # torch.cuda.synchronize()
                # forward_start = time.time()
                output = model(input)
                # if args.rank == 0:
                #    print("forward_fininshed in {} ".format(time.time() - forward_start))

            # --- calculate loss function(s) ---
            loss = loss_fn(output, target) / args.iters_to_accumulate

            # exit if loss is NaN
            if torch.isnan(loss):
                _logger.info('Loss is NaN')
        if not args.distributed:
            losses_m.update(loss.item() * args.iters_to_accumulate, input.size(0))

        if loss_scaler is not None:
            # torch.distributed.barrier()
            # torch.cuda.synchronize()
            # backward_start = time.time()
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order,
                batch_idx=batch_idx,
                iters_to_accumulate=args.iters_to_accumulate)
            #
            # if args.rank==0:
            #    print("bakward_fininshed in {} ".format(time.time() - backward_start))
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                utils.dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            if (batch_idx + 1) % args.iters_to_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            lr_str = " | ".join([f"{lr_i:.3e}" for i, lr_i in enumerate(lrl)])

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data * args.iters_to_accumulate, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.rank == 0:
                eta_seconds = batch_time_m.avg * (len(loader) - batch_idx)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  |  '
                    'ETA: {eta}  |  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  |  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  |  '
                    'LR: [{lr_str}] ({lr:.3e})  |  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})  |  '
                    'Max Mem: {memory:.0f}'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        lr_str=lr_str,
                        data_time=data_time_m,
                        memory=torch.cuda.max_memory_allocated() / MB,
                        eta=eta_string,
                    ))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if (batch_idx + 1) % args.iters_to_accumulate == 0:
            num_updates += 1
            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    train_stats = OrderedDict(
        [('loss', losses_m.avg)] +
        [(f"lr_param_group_{i}", param_group['lr']) for i, param_group in enumerate(optimizer.param_groups)]
    )

    return train_stats


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if "mapet" in args.model:
                input, _, target, mask_query, mask_content, index_vector, upper_predictions = data
                target_selection = index_vector * upper_predictions
                target = target[target_selection[:, 1:]].reshape(input.shape[0], -1).long()
            else:
                input, target = data

            if DEBUG and batch_idx == DEBUG_BATCHES:
                break

            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
                if "mapet" in args.model:
                    mask_query, mask_content = mask_query.cuda(), mask_content.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():

                if "mapet" in args.model:
                    output = model(input, mask_query, mask_content, index_vector)
                    output = output.permute(1, 0, 2)[:,
                             upper_predictions[index_vector].reshape(input.shape[0], -1)].reshape(8192, input.shape[0], -1).permute(1, 0, 2)
                else:
                    output = model(input)

                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                loss = loss_fn(output, target)

            if "mapet" in args.model:
                acc1 = accuracy_sequence(output, target, (args.num_tokens + 1))
                # acc1 = torch.tensor(acc1).cuda()
                acc5 = torch.zeros(1).cuda()
            else:
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                eta_seconds = batch_time_m.avg * (len(loader) - batch_idx)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'ETA: {eta}  |  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m, eta=eta_string))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


# From timm, problem in the gestion of sequence prediction like xlnet
def accuracy_sequence(output, target, non_predict_value=8193):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    pred_size = torch.count_nonzero(target != (non_predict_value))
    _, preds = torch.max(output, 1)
    running_corrects = torch.sum(preds == target)
    acc = running_corrects.double() / pred_size
    return acc
