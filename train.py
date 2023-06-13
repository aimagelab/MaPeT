#!/usr/bin/env python3
""" Training Script
"""
import json
import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy
from timm.models import create_model, safe_model_name, load_checkpoint, convert_splitbn_model, \
    convert_sync_batchnorm, set_fast_norm, resume_checkpoint
# from utils.torch_load import resume_checkpoint
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as NativeDDP

import utils.distributed as distributed
from arguments_manager import get_args_parser, _parse_args
from data.clipped_imagenet import SequenceDataset
from engine import train_one_epoch, validate
from models import *
from utils.cuda_amp import ApexScaler, NativeScaler
from utils.logger import init_wandb_logger, custom_setup_default_logging

# from fvcore.nn import FlopCountAnalysis
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError as e:
    has_functorch = False

_logger = logging.getLogger('train')


@record
def main():
    parser, config_parser = get_args_parser()
    args, args_text = _parse_args(parser, config_parser)

    args.prefetcher = not args.no_prefetcher
    args.deterministic = not args.no_deterministic

    distributed.init_distributed_device(args)

    if not args.distributed:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    distributed.set_random_seed(args.seed, deterministic=args.deterministic, benchmark=args.benchmark)
    multi_process_cfg = {
        "num_workers": args.workers,
        "mp_start_method": None,
        "opencv_num_threads": None,
        "omp_num_threads": None,
        "mkl_num_threads": None,
    }
    distributed.setup_multi_processes(multi_process_cfg)

    dash_line = '-' * 60 + '\n'
    output_dir = Path(args.output)
    if args.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        custom_setup_default_logging(log_path=os.path.join(args.output, "log.txt"))
        env_var = sorted([(k, v) for k, v in os.environ.items()], key=(lambda x: x[0]))
        # log env variables
        env_info = '\n'.join([f'-{k}: {v}' for k, v in env_var])
        _logger.info('Environment info:\n' + env_info + '\n' + dash_line)
        # log args
        args_info = '\n'.join([f'-{k}: {v}' for k, v in sorted(vars(args).items())])
        _logger.info(dash_line + '\nArgs info:\n' + args_info + '\n' + dash_line)
        if args.log_wandb:
            _logger.info(f">>> Setup wandb: {'ON' if args.log_wandb else 'OFF'}")
            if has_wandb:
                init_wandb_logger(args)
            else:
                _logger.warning("You've requested to log metrics to wandb but package not found. "
                                "Metrics not being logged to wandb, try `pip install wandb`")

    if not args.resume and os.path.exists(output_dir / "last.pth.tar"):
        args.resume = str(output_dir / "last.pth.tar")

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDIA apex or upgrade to PyTorch 1.6")

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    # Add custom arguments for custom models
    kwargs = {}
    if args.pre_training_checkpoint:
        kwargs["pre_training_checkpoint"] = args.pre_training_checkpoint
    train_kwargs = {}  # kwargs used in the training loop
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        init_values=args.layer_scale_init_value,
        **kwargs
    )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.rank == 0:
        if args.log_wandb:
            wandb.run.summary["#Params_total"] = n_params_total
            wandb.run.summary["#Params_requires_grad"] = n_params_grad
        _logger.info(
            f'Model {safe_model_name(args.model)} created:'
            f'\n\t-#Params total:         {n_params_total}'
            f'\n\t-#Params requires_grad: {n_params_grad}'
        )

    data_config = resolve_data_config(vars(args), model=model, verbose=args.rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ''  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if args.rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)
    if args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if args.patch_embed_lr is not None:
        my_list = 'patch_embed'
        params_patch_embed = [v for k, v in model.named_parameters() if my_list in k]
        params_model = [v for k, v in model.named_parameters() if my_list not in k]
        parameters = [
            {'params': params_model},
            {'params': params_patch_embed, 'lr': args.patch_embed_lr},
        ]
        optimizer = create_optimizer_v2(parameters, **optimizer_kwargs(cfg=args))
    else:
        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(
                model,
                device_ids=[args.local_rank],
                broadcast_buffers=not args.no_ddp_bb,
                find_unused_parameters=False,
            )
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval datasets
    if "mapet" in args.model:
        dataset_train = SequenceDataset(
            args.kmeans_train,
            args.csv_train,
            sequence_len=model.module.embed_len,
            num_attention_heads=model.module.num_heads,
            least_number=args.least_number,
            isDir=True,
            pre_trained=False,
            positional=False,
            root_path=args.root_dir,
            mapet=True
        )
        dataset_eval = SequenceDataset(
            args.kmeans_test,
            args.csv_test,
            sequence_len=model.module.embed_len,
            num_attention_heads=model.module.num_heads,
            least_number=args.least_number,
            isDir=False,
            pre_trained=False,
            positional=False,
            root_path=args.root_dir,
            mapet=True
        )
    else:
        dataset_train = create_dataset(
            args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats)
        dataset_eval = create_dataset(
            args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size)

    if args.rank == 0 and hasattr(dataset_train, "parser") and hasattr(dataset_eval, "parser"):
        _logger.info(f'Saving train class mapping')
        with open(f'{output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")}_train_class_mapping.json', 'w') as fp:
            json.dump(dataset_train.parser.class_to_idx, fp)
        _logger.info(f'Saving val class mapping')
        with open(f'{output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")}_val_class_mapping.json', 'w') as fp:
            json.dump(dataset_eval.parser.class_to_idx, fp)

        assert dataset_train.parser.class_to_idx == dataset_eval.parser.class_to_idx, "Class to index must be the same for train and val"

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
        persistent_workers=args.workers > 0,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        persistent_workers=args.workers > 0,

    )
    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    if args.rank == 0:
        decreasing = True if eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    # if args.rank == 0:
    #     input_model = torch.ones([1,3,224,224]).cuda()
    #     flops = FlopCountAnalysis(model, input_model)
    #     flops.total()
    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)
            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema,
                mixup_fn=mixup_fn, **train_kwargs,
            )

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(
                model,
                loader_eval,
                validate_loss_fn,
                args,
                amp_autocast=amp_autocast,
                log_suffix=f" {epoch} ",
            )

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast,
                    log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if args.rank == 0 and output_dir is not None:
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
                if args.log_wandb:
                    wandb.run.summary[f"best_{eval_metric}"] = best_metric
                    wandb.run.summary[f"best_epoch"] = best_epoch

    except KeyboardInterrupt:
        pass
    except ValueError as e:
        _logger.info(e)
        # terminate distributed processes
        if args.distributed:
            dist.destroy_process_group()
        raise

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


if __name__ == '__main__':
    if 'CONDA_DEFAULT_ENV' in os.environ:
        print(f"Conda venv: {os.environ['CONDA_DEFAULT_ENV']}")
    main()
