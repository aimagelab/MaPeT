import argparse

import yaml


def _parse_args(parser, config_parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def get_args_parser():
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser(description='PyTorch Training')

    # ### Custom ###
    parser.add_argument('--distributed', default=True, type=bool, help='distributed training?')
    parser.add_argument('--no_deterministic', action="store_true", help='deterministic results?')
    # wandb setup
    parser.add_argument("--wandb_logging", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", default="memory-vit", type=str)
    parser.add_argument("--wandb_entity", default="lorenzo_b_master_thesis", type=str)
    parser.add_argument("--wandb_name", default=None, type=str)
    parser.add_argument("--wandb_id", default=None, type=str)
    parser.add_argument(
        "--wandb_resume", default="allow", choices=["never", "must", "allow"], type=str
    )
    parser.add_argument("--wandb_group", default=None, type=str)
    parser.add_argument("--wandb_notes", default=None, type=str)

    # Memory-Vit
    parser.add_argument("--samples_per_centroids", type=int, default=256)
    parser.add_argument("--n_memory_slots", type=int, default=64)
    parser.add_argument("--deque_iters", type=int, default=10)
    parser.add_argument("--window", type=float, default=0.25)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--add_memory_attention", action="store_true")
    parser.add_argument("--use_global_visual_in_memory", action="store_true")
    parser.add_argument("--use_parallel_attention", action="store_true")
    parser.add_argument("--use_attn_outputs_as_memory_query", action="store_true")
    parser.add_argument("--use_gated_memory", action="store_true")
    parser.add_argument("--add_memory_slots_selfattn", action="store_true")
    parser.add_argument("--disable_IP_to_L2", action="store_true")
    parser.add_argument("--kmeans_memory", action="store_true")
    parser.add_argument("--head_kmeans_memory", action="store_true")
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--kmeans_memory_with_gradient", action="store_true")
    parser.add_argument("--freeze_memory", action="store_true")
    parser.add_argument("--clean_memory_at_first_step", action="store_true")
    parser.add_argument("--retrieval_memory", action="store_true")
    parser.add_argument("--create_index_on_gpu", action="store_true")
    parser.add_argument("--init_centroids_with_old_mems", action="store_true")

    # Clipped_ViT
    parser.add_argument("--double_clipped_losses", action="store_true")
    parser.add_argument("--patch_embed_loss", default="MAE", type=str)
    parser.add_argument("--clipped_patch_embed_weights", type=str, default="",
                        help="Path to checkpoint of Clipped PatchEmbed")
    parser.add_argument("--patch_embed_lr", default=None, type=float,
                        help="different learning rate for the patch embedding")
    parser.add_argument('--pre_training_checkpoint', default='',
                        help='checkpoint path of pre-training parameters')
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Size of the square patches"
    )
    parser.add_argument(
        "--infonce_loss_temperature", default=0.2, type=float, help="Temperature for InfoNCE loss"
    )
    parser.add_argument(
        "--least_number",
        default=50,
        type=int,
        help="Minimum number of element in the permutation needed before predicting",
    )
    parser.add_argument(
        "--upper_limit",
        default=197,
        type=int,
        help="Maximum number of elments to be predicted",
    )

    # kmeans args
    parser.add_argument(
        "--num_tokens",
        default=8192,
        type=int,
        help="number of possible index outputted by kmeans",
    )
    parser.add_argument(
        "--embed_dim",
        default=192,
        type=int,
        help="Embed dimension of vision transformer",
    )

    # ##############

    # Dataset parameters
    group = parser.add_argument_group('Dataset parameters')
    # Keep this argument outside of the dataset group because it is positional.
    parser.add_argument('data_dir', metavar='DIR',
                        help='path to dataset')
    group.add_argument('--dataset', '-d', metavar='NAME', default='',
                       help='dataset type (default: ImageFolder/ImageTar if empty)')
    group.add_argument('--kmeans_train', metavar='NAME',
                       default='PATH/TO/Kmeans_centroids/KMEANS/imnet_8192',
                       help='path of the imagenet train processed by clip and discretized')
    group.add_argument('--kmeans_test',
                       default='PATH/TO/Kmeans_centroids/KMEANS/k_means_processed_imagenet_8192_val.pth',
                       help='path of the imagenet test processed by clip and discretized')
    group.add_argument('--csv_train', default='PATH/TO/Imagenet_labels_train_new.csv',
                       help='path of the dataset train csv')
    group.add_argument('--csv_test', default='PATH/TO/Imagenet_labels_test_new.csv',
                       help='path of the dataset test csv')
    group.add_argument('--root_dir', default='PATH/TO',
                       help='Path to add to a csv file')
    group.add_argument('--train-split', metavar='NAME', default='train',
                       help='dataset train split (default: train)')
    group.add_argument('--val-split', metavar='NAME', default='validation',
                       help='dataset validation split (default: validation)')
    group.add_argument('--dataset-download', action='store_true', default=False,
                       help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                       help='path to class to idx mapping file (default: "")')
    group.add_argument('--patch_augmentation', action='store_true', default=False,
                       help='If use the augmentation  also at patch level')
    group.add_argument('--blockwise_perm', action='store_true', default=False,
                       help='If use the blockwise permutation')
    # ##############
    # BeiT
    parser.add_argument('--num_mask_patches', default=75, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)
    # BerT
    parser.add_argument('--randomize_masking', default=False, action='store_true',
                        help='if setted random masking is applied instead of block masking')
    # CAE
    parser.add_argument('--regressor_depth', default=4, type=int, help='depth of the regressor')
    parser.add_argument('--decoder_depth', default=4, type=int, help='depth of the decoder')
    # parser.add_argument('--decoder_embed_dim', default=768, type=int,
    #                     help='dimensionaltiy of embeddings for decoder')
    parser.add_argument('--decoder_embed_dim', default=192, type=int,
                        help='dimensionaltiy of embeddings for decoder')
    parser.add_argument('--decoder_num_heads', default=12, type=int,
                        help='Number of heads for decoder')
    parser.add_argument('--decoder_num_classes', default=8192, type=int,
                        help='Number of classes for decoder')
    parser.add_argument('--decoder_layer_scale_init_value', default=0, type=float,
                        help='decoder layer scale init value')
    parser.add_argument('--layer_scale_init_value', type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    # alignment constraint
    parser.add_argument('--align_loss_weight', type=float, default=2, help='loss weight for the alignment constraint')

    # Model parameters
    group = parser.add_argument_group('Model parameters')
    group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                       help='Name of model to train (default: "resnet50"')
    group.add_argument('--pretrained', action='store_true', default=False,
                       help='Start with pretrained version of specified network (if avail)')
    group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                       help='Initialize model from this checkpoint (default: none)')
    group.add_argument('--resume', default='', type=str, metavar='PATH',
                       help='Resume full model and optimizer state from checkpoint (default: none)')
    group.add_argument('--no-resume-opt', action='store_true', default=False,
                       help='prevent resume of optimizer state when resuming model')
    group.add_argument('--num-classes', type=int, default=None, metavar='N',
                       help='number of label classes (Model default if None)')
    group.add_argument('--gp', default=None, type=str, metavar='POOL',
                       help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    group.add_argument('--img-size', type=int, default=None, metavar='N',
                       help='Image patch size (default: None => model default)')
    group.add_argument('--input-size', default=None, nargs=3, type=int,
                       metavar='N N N',
                       help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    group.add_argument('--crop-pct', default=None, type=float,
                       metavar='N', help='Input image center crop percent (for validation only)')
    group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                       help='Override mean pixel value of dataset')
    group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                       help='Override std deviation of dataset')
    group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                       help='Image resize interpolation type (overrides model)')
    group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                       help='Input batch size for training (default: 128)')
    group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                       help='Validation batch size override (default: None)')
    group.add_argument('--channels-last', action='store_true', default=False,
                       help='Use channels_last memory layout')
    scripting_group = group.add_mutually_exclusive_group()
    scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                                 help='torch.jit.scripts the full model')
    scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                                 help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)")
    group.add_argument('--fuser', default='', type=str,
                       help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
    group.add_argument('--fast-norm', default=False, action='store_true',
                       help='enable experimental fast-norm')
    group.add_argument('--grad-checkpointing', action='store_true', default=False,
                       help='Enable gradient checkpointing through model blocks/stages')

    # Optimizer parameters
    group = parser.add_argument_group('Optimizer parameters')
    group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                       help='Optimizer (default: "sgd"')
    group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                       help='Optimizer Epsilon (default: None, use opt default)')
    group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                       help='Optimizer Betas (default: None, use opt default)')
    group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                       help='Optimizer momentum (default: 0.9)')
    group.add_argument('--weight-decay', type=float, default=2e-5,
                       help='weight decay (default: 2e-5)')
    group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                       help='Clip gradient norm (default: None, no clipping)')
    group.add_argument('--clip-mode', type=str, default='norm',
                       help='Gradient clipping mode. One of ("norm", "value", "agc")')
    group.add_argument('--layer-decay', type=float, default=None,
                       help='layer-wise learning rate decay (default: None)')

    # Learning rate schedule parameters
    group = parser.add_argument_group('Learning rate schedule parameters')
    group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                       help='LR scheduler (default: "step"')
    group.add_argument('--lr', type=float, default=0.05, metavar='LR',
                       help='learning rate (default: 0.05)')
    group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                       help='learning rate noise on/off epoch percentages')
    group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                       help='learning rate noise limit percent (default: 0.67)')
    group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                       help='learning rate noise std-dev (default: 1.0)')
    group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                       help='learning rate cycle len multiplier (default: 1.0)')
    group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                       help='amount to decay each learning rate cycle (default: 0.5)')
    group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                       help='learning rate cycle limit, cycles enabled if > 1')
    group.add_argument('--lr-k-decay', type=float, default=1.0,
                       help='learning rate k-decay for cosine/poly (default: 1.0)')
    group.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                       help='warmup learning rate (default: 0.0001)')
    group.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                       help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    group.add_argument('--epochs', type=int, default=300, metavar='N',
                       help='number of epochs to train (default: 300)')
    group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                       help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                       help='manual epoch number (useful on restarts)')
    group.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                       help='list of decay epoch indices for multistep lr. must be increasing')
    group.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                       help='epoch interval to decay LR')
    group.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                       help='epochs to warmup LR, if scheduler supports')
    group.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                       help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                       help='patience epochs for Plateau LR scheduler (default: 10')
    group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                       help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    group = parser.add_argument_group('Augmentation and regularization parameters')
    group.add_argument('--no-aug', action='store_true', default=False,
                       help='Disable all training augmentation, override other train aug args')
    group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                       help='Random resize scale (default: 0.08 1.0)')
    group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                       help='Random resize aspect ratio (default: 0.75 1.33)')
    group.add_argument('--hflip', type=float, default=0.5,
                       help='Horizontal flip training aug probability')
    group.add_argument('--vflip', type=float, default=0.,
                       help='Vertical flip training aug probability')
    group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                       help='Color jitter factor (default: 0.4)')
    group.add_argument('--aa', type=str, default=None, metavar='NAME',
                       help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    group.add_argument('--aug-repeats', type=float, default=0,
                       help='Number of augmentation repetitions (distributed training only) (default: 0)')
    group.add_argument('--aug-splits', type=int, default=0,
                       help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    group.add_argument('--jsd-loss', action='store_true', default=False,
                       help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    group.add_argument('--bce-loss', action='store_true', default=False,
                       help='Enable BCE loss w/ Mixup/CutMix use.')
    group.add_argument('--bce-target-thresh', type=float, default=None,
                       help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                       help='Random erase prob (default: 0.)')
    group.add_argument('--remode', type=str, default='pixel',
                       help='Random erase mode (default: "pixel")')
    group.add_argument('--recount', type=int, default=1,
                       help='Random erase count (default: 1)')
    group.add_argument('--resplit', action='store_true', default=False,
                       help='Do not random erase first (clean) augmentation split')
    group.add_argument('--mixup', type=float, default=0.0,
                       help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix', type=float, default=0.0,
                       help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                       help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    group.add_argument('--mixup-prob', type=float, default=1.0,
                       help='Probability of performing mixup or cutmix when either/both is enabled')
    group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                       help='Probability of switching to cutmix when both mixup and cutmix enabled')
    group.add_argument('--mixup-mode', type=str, default='batch',
                       help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                       help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    group.add_argument('--smoothing', type=float, default=0.1,
                       help='Label smoothing (default: 0.1)')
    group.add_argument('--train-interpolation', type=str, default='random',
                       help='Training interpolation (random, bilinear, bicubic default: "random")')
    group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                       help='Dropout rate (default: 0.)')
    group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                       help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                       help='Drop path rate (default: None)')
    group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                       help='Drop block rate (default: None)')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    group = parser.add_argument_group('Batch norm parameters',
                                      'Only works with gen_efficientnet based models currently.')
    group.add_argument('--bn-momentum', type=float, default=None,
                       help='BatchNorm momentum override (if not None)')
    group.add_argument('--bn-eps', type=float, default=None,
                       help='BatchNorm epsilon override (if not None)')
    group.add_argument('--sync-bn', action='store_true',
                       help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    group.add_argument('--dist-bn', type=str, default='reduce',
                       help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    group.add_argument('--split-bn', action='store_true',
                       help='Enable separate BN layers per augmentation split.')

    # Model Exponential Moving Average
    group = parser.add_argument_group('Model exponential moving average parameters')
    group.add_argument('--model-ema', action='store_true', default=False,
                       help='Enable tracking moving average of model weights')
    group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                       help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    group.add_argument('--model-ema-decay', type=float, default=0.9998,
                       help='decay factor for model weights moving average (default: 0.9998)')

    # Misc
    group = parser.add_argument_group('Miscellaneous parameters')
    group.add_argument('--seed', type=int, default=42, metavar='S',
                       help='random seed (default: 42)')
    group.add_argument('--benchmark', action='store_true', default=False)
    group.add_argument('--worker-seeding', type=str, default='all',
                       help='worker seed mode (default: all)')
    group.add_argument('--log-interval', type=int, default=50, metavar='N',
                       help='how many batches to wait before logging training status')
    group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                       help='how many batches to wait before writing recovery checkpoint')
    group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                       help='number of checkpoints to keep (default: 10)')
    group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                       help='how many training processes to use (default: 4)')
    group.add_argument('--save-images', action='store_true', default=False,
                       help='save images of input bathes every log interval for debugging')
    group.add_argument('--amp', action='store_true', default=False,
                       help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    group.add_argument('--apex-amp', action='store_true', default=False,
                       help='Use NVIDIA Apex AMP mixed precision')
    group.add_argument('--native-amp', action='store_true', default=False,
                       help='Use Native Torch AMP mixed precision')
    group.add_argument('--iters_to_accumulate', default='1', type=int,
                       help='number of batches evaluated before performing an optimizer step. Used for Gradient accumulation')
    group.add_argument('--no-ddp-bb', action='store_true', default=False,
                       help='Force broadcast buffers for native DDP to off.')
    group.add_argument('--pin-mem', action='store_true', default=False,
                       help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    group.add_argument('--no-prefetcher', action='store_true', default=False,
                       help='disable fast prefetcher')
    group.add_argument('--output', default='', type=str, metavar='PATH', required=True,
                       help='path to output folder (default: none, current dir)')
    group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                       help='Best metric (default: "top1"')
    group.add_argument('--tta', type=int, default=0, metavar='N',
                       help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    group.add_argument("--local_rank", default=0, type=int)
    group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                       help='use the multi-epochs-loader to save time at the beginning of every epoch')
    group.add_argument('--log-wandb', action='store_true', default=False,
                       help='log training and validation metrics to wandb')

    return parser, config_parser
