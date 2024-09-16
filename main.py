import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pruner import init_pruner

from args import get_args
from utils import *
from datasets import *
from models import *
from nipq.nipq import QuantOps as Q
from nipq.train_test import (
    resume_checkpoint,
    create_checkpoint,
    test,
    train_aux,
    # train_aux_target_avgbit,
    train_aux_target_bops,
    CosineWithWarmup,
    bops_cal,
    bit_loss,
    accuracy,
    categorize_param,
    bit_cal,
    get_optimizer
)
args = get_args()

seed_everything(args.seed)

# model
if args.model == 'DKN':
    model = DKN(kernel_size=3, filter_size=15, residual=True)
elif args.model == 'FDKN':
    model = FDKN(kernel_size=3, filter_size=15, residual=True)
elif args.model == 'DJF':
    model = DJF(residual=True)
elif args.model == 'JIIF':
    model = JIIF(args, 128, 128)

else:
    raise NotImplementedError(f'Model {args.model} not found')

# loss
if args.loss == 'L1':
    criterion = nn.L1Loss()
elif args.loss == 'L2':
    criterion = nn.MSELoss()
else:
    raise NotImplementedError(f'Loss {args.loss} not found')

# dataset
if args.dataset == 'NYU':
    dataset = NYUDataset
elif args.dataset == 'Lu':
    dataset = LuDataset
elif args.dataset == 'Middlebury':
    dataset = MiddleburyDataset
elif args.dataset == 'NoisyMiddlebury':
    dataset = NoisyMiddleburyDataset
else:
    raise NotImplementedError(f'Dataset {args.loss} not found')

if args.model in ['JIIF']:
    if not args.test:
        train_dataset = dataset(root=args.data_root, split='train', scale=args.scale, downsample=args.interpolation,
                                augment=True, to_pixel=True, sample_q=args.sample_q, input_size=args.input_size, noisy=args.noisy)
    test_dataset = dataset(root=args.data_root, split='test', scale=args.scale,
                           downsample=args.interpolation, augment=False, to_pixel=True, sample_q=None)  # full image
elif args.model in ['DJF', 'DKN', 'FDKN']:
    if not args.test:
        train_dataset = dataset(root=args.data_root, split='train', scale=args.scale, downsample=args.interpolation,
                                augment=True, pre_upsample=True, input_size=args.input_size, noisy=args.noisy)
    test_dataset = dataset(root=args.data_root, split='test', scale=args.scale,
                           downsample=args.interpolation, augment=False, pre_upsample=True)
else:
    raise NotImplementedError(f'Dataset for model type {args.model} not found')

# dataloader
if not args.test:
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch, pin_memory=True, drop_last=False, shuffle=True, num_workers=args.num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch,
                                          pin_memory=True, drop_last=False, shuffle=False, num_workers=args.num_workers)


if args.pruning:
    if not args.resume and args.checkpoint:
        checkpoint_dict = torch.load(args.checkpoint, map_location='cuda')
        state_dict = checkpoint_dict['model']
        model_dict = model.state_dict()

        model_keys = model_dict.keys()
        state_keys = list(state_dict.keys())

        state_idx = 0
        for key in model_keys:
            if 'weight' in key or 'bias' in key:
                assert (model_dict[key].shape ==
                        state_dict[state_keys[state_idx]].shape)
                model_dict[key] = state_dict[state_keys[state_idx]].clone()
                state_idx += 1

        model.load_state_dict(model_dict)

        print('LOAD CHECKPOINT')

    # args.pruner : str(KETIPrunerStructured), wgt(KETIPrunerWeight)
    pruner = init_pruner(model, args, None)
    model = pruner.prune()

    for n, m in model.named_modules():
        if hasattr(m, 'mask'):
            mask = m.mask == 0
            print(
                n, m.weight[mask.to(m.weight.device).view_as(m.weight)].sum())


if args.quantize:
    replace_module(model, last_fp=args.last_fp)
    if args.mode == 'avgbit':
        print(f'** act_q : {args.a_scale} / weight_q : {args.w_scale}')
    elif args.mode == 'bops':
        print(f'** bops scale : {args.bops_scale}')
    else:
        raise NotImplementedError()

    if not args.resume and args.checkpoint:
        checkpoint_dict = torch.load(args.checkpoint, map_location='cuda')
        state_dict = checkpoint_dict['model']
        model_dict = model.state_dict()

        model_keys = model_dict.keys()
        state_keys = list(state_dict.keys())

        state_idx = 0
        for key in model_keys:
            if 'weight' in key or 'bias' in key:
                assert (model_dict[key].shape ==
                        state_dict[state_keys[state_idx]].shape)
                model_dict[key] = state_dict[state_keys[state_idx]].clone()
                state_idx += 1

        model.load_state_dict(model_dict)
        print('LOAD CHECKPOINT')

    mask_dict = checkpoint_dict.get('mask')
    if mask_dict is not None:
        for n, m in model.named_modules():
            if hasattr(m, 'mask'):
                print('set mask for', n)
                m.mask = mask_dict[f'{n}.mask']
                print(m.mask.device)

# trainer
if not args.test:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    trainer = Trainer(args, args.name, model, objective=criterion, optimizer=optimizer, lr_scheduler=scheduler, metrics=[
                      RMSEMeter(args)], device='cuda', use_checkpoint='scratch' if ((args.quantize or args.pruning) and not args.resume) else args.checkpoint, eval_interval=args.eval_interval, workspace=f'workspace/{args.exp}' if args.exp else 'workspace')
else:
    trainer = Trainer(args, args.name, model, objective=criterion, metrics=[
                      RMSEMeter(args)], device='cuda', use_checkpoint='scratch' if ((args.quantize or args.pruning) and not args.resume) else args.checkpoint)


if args.quantize:
    Q.initialize(model, act=args.a_scale > 0, weight=args.w_scale > 0)

    def forward_hook(module, inputs, outputs):
        module.out_shape = outputs.shape

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (Q.ReLU, Q.Sym, Q.HSwish)):
            hooks.append(module.register_forward_hook(forward_hook))

        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(forward_hook))
    img = torch.Tensor(1, 3, args.input_size, args.input_size).cuda()

    for data in train_loader:
        break
    data = trainer.prepare_data(data)
    model.eval()
    model.cuda()
    model(data)

    weight, bnbias = categorize_param(model)

    if args.optim == 'adam':
        optimizer = optim.Adam(params=[
            {'params': bnbias, 'weight_decay': 0., 'lr': args.lr},
            {'params': weight, 'weight_decay': 1e-5, 'lr': args.lr},
        ], lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(params=[
            {'params': bnbias, 'weight_decay': 0., 'lr': args.lr},
            {'params': weight, 'weight_decay': 1e-5, 'lr': args.lr},
        ], lr=args.lr, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    model_ema = None
    model_t = None
    if args.resume:
        start_epoch = trainer.epoch + 1
    else:
        start_epoch = 0
    for epoch in range(start_epoch, int(args.epoch)):
        print('Epoch', epoch)
        trainer.epoch = epoch
        train_acc, train_loss, _ = trainer.train_aux_target_avgbit(train_loader, epoch, ema_rate=0.99, bit_scale_a=args.a_scale, bit_scale_w=args.w_scale,
                                                                   target_ours=args.target, scale=args.bops_scale)

        # train_regular(train_loader, model, model_ema, None, 0, 0, criterion, optimizer, epoch, [1.,], ema_rate=0.9997)
        # acc_base, test_loss = test(test_loader, model, criterion, epoch, False)
        acc_ema = 0
        if model_ema is not None:
            acc_ema, loss_ema = test(test_loader, model_ema, criterion, epoch)

        a_bit, au_bit, w_bit, wu_bit = bit_cal(model)
        bops_total = bops_cal(model)
        print(
            f'Epoch : [{epoch}] / a_bit : {au_bit}bit / w_bit : {wu_bit}bit / bops : {bops_total.item()}GBops')

        if epoch % args.eval_interval == 0:
            trainer.evaluate_one_epoch(test_loader)
            trainer.save_checkpoint(full=False, best=True)
        else:
            trainer.save_checkpoint(full=False, best=False)

     # BN tuning phase
    Q.initialize(model, act=args.a_scale > 0,
                 weight=args.w_scale > 0, noise=False)

    for name, module in model.named_modules():
        if isinstance(module, (Q.ReLU, Q.Sym, Q.HSwish, Q.Conv2d, Q.Linear)):
            module.bit.requires_grad = False

    print('Finetuning')
    trainer.epoch += 1
    trainer.eval_interval = 1
    trainer.train(train_loader, test_loader, args.epoch + args.ft_epoch)

else:
    # main
    if not args.test:
        trainer.train(train_loader, test_loader, args.epoch)

    if args.save:
        # save results (doesn't need GT)
        trainer.test(test_loader)
    else:
        # evaluate (needs GT)
        trainer.evaluate(test_loader)
        # pass
