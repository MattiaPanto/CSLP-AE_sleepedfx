import sys
from split_model import SplitLatentModel
import torch
import numpy as np
import torch.optim as optim
from extra.utils import get_results, get_eval_results, get_split_latents, split_do_tsne, plot_latents, SleepCustomLoader, fit_knn_fn, fit_etc_fn
#from utils import SleepCustomLoader
from conversion_utils import get_full_conversion_results
from sklearn.decomposition import PCA
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import os
import pandas as pd
import argparse

from sklearn.model_selection import GroupKFold, KFold, train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='/root/data/sleepedf_dataset/')
parser.add_argument('--model_save_dir', type=str, default='./')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=50) #50
parser.add_argument('--lr', type=float, default=0.0001)

parser.add_argument('--channels', type=int, default=200) # 200
parser.add_argument('--in_channels', type=int, default=50) #50
parser.add_argument('--num_layers', type=int, default=3) #3
parser.add_argument('--latent_dim', type=int, default=40) #40
parser.add_argument('--recon_type', type=str, default='mse')
parser.add_argument('--content_cosine', type=int, default=1)

parser.add_argument('--csv_file', type=str, default='data_labels.csv') #sleepedfx

parser.add_argument('--final_div_factor', type=int, default=10)
parser.add_argument('--initial_lr', type=float, default=0.0001)
parser.add_argument('--max_lr', type=float, default=0.0001)
parser.add_argument('--pct_start', type=float, default=0.5)

parser.add_argument('--sub_cross_s_enabled', type=int, default=0)
parser.add_argument('--sub_cross_s_weight', type=float, default=1.0)
parser.add_argument('--task_cross_t_enabled', type=int, default=0)
parser.add_argument('--task_cross_t_weight', type=float, default=1.0)

parser.add_argument('--recon_enabled', type=int, default=0)
parser.add_argument('--recon_weight', type=float, default=1.0)

parser.add_argument('--scramble_permute_enabled', type=int, default=0)
parser.add_argument('--scramble_permute_weight', type=float, default=1.0)

parser.add_argument('--conversion_permute_enabled', type=int, default=0)
parser.add_argument('--conversion_permute_weight', type=float, default=1.0)

parser.add_argument('--quadruplet_permute_enabled', type=int, default=0)
parser.add_argument('--quadruplet_permute_F_enabled', type=int, default=0)
parser.add_argument('--quadruplet_permute_weight', type=float, default=1.0)

parser.add_argument('--sub_contra_s_enabled', type=int, default=1)
parser.add_argument('--sub_contra_s_weight', type=float, default=1.0)
parser.add_argument('--task_contra_t_enabled', type=int, default=1)
parser.add_argument('--task_contra_t_weight', type=float, default=1.0)

parser.add_argument('--latent_permute_s_enabled', type=int, default=1)
parser.add_argument('--latent_permute_s_weight', type=float, default=1.0)
parser.add_argument('--latent_permute_t_enabled', type=int, default=1)
parser.add_argument('--latent_permute_t_weight', type=float, default=1.0)

parser.add_argument('--restored_permute_s_enabled', type=int, default=0)
parser.add_argument('--restored_permute_s_weight', type=float, default=1.0)
parser.add_argument('--restored_permute_t_enabled', type=int, default=0)
parser.add_argument('--restored_permute_t_weight', type=float, default=1.0)

parser.add_argument('--sub_content_enabled', type=int, default=0)
parser.add_argument('--sub_content_weight', type=float, default=1.0)
parser.add_argument('--task_content_enabled', type=int, default=0)
parser.add_argument('--task_content_weight', type=float, default=1.0)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--use_tqdm', type=int, default=1)

parser.add_argument('--group', type=str, default='')

parser.add_argument('--override_seed', type=int, default=None)

parser.add_argument('--extra_tags', type=str, default='')

parser.add_argument('--full_eval', type=int, default=1)

parser.add_argument('--eval_every', type=int, default=70)
parser.add_argument('--sched_mode', type=str, default='max')
parser.add_argument('--sched_patience', type=int, default=5)
parser.add_argument('--sched_factor', type=float, default=0.5)
parser.add_argument('--old_sched', type=int, default=1)
parser.add_argument('--save_model', type=int, default=1)
parser.add_argument('--add_name', type=str, default='')
parser.add_argument('--conversion_N', type=int, default=2000)
parser.add_argument('--extra_classifiers', type=int, default=1)
parser.add_argument('--conversion_results', type=int, default=1)

args, unknown = parser.parse_known_args()

loss_to_notation = {
    'recon': ['R'],
    'sub_contra_s': ['SL', 'CR:s'],
    'task_contra_t': ['SL', 'CR:t'],
    'latent_permute_s': ['SL', 'LP:s'],
    'latent_permute_t': ['SL', 'LP:t'],
    'restored_permute_s': ['SL', 'RP:s'],
    'restored_permute_t': ['SL', 'RP:t'],
    'sub_content': ['SL', 'C:s'],
    'task_content': ['SL', 'C:t'],
    'sub_cross_s': ['CE:s'],
    'task_cross_t': ['CE:t'],
    'scramble_permute': ['SP'],
    'conversion_permute': ['CP'],
    'quadruplet_permute': ['QP'],
    'quadruplet_permute_f': ['QPf'],
}

if __name__ == '__main__':
    print(args, file=sys.stdout, flush=True)
    if args.random_seed:
        SEED = np.random.randint(0, 2**31 - 1)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    elif args.override_seed is not None:
        SEED = args.override_seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    else:
        SEED = 3242342323
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    IN_CHANNELS = args.in_channels
    NUM_LAYERS = args.num_layers
    KERNEL_SIZE = 4
    
    USE_TQDM = args.use_tqdm
    
    OLD_SCHED = bool(args.old_sched)

    all_losses = ["recon", "sub_contra_s", "task_contra_t", "latent_permute_s", "latent_permute_t", "restored_permute_s", "restored_permute_t", "sub_content", "task_content", "sub_cross_s", "task_cross_t", "scramble_permute", "conversion_permute", "quadruplet_permute", "quadruplet_permute_F"]
    losses = []
    loss_weights = defaultdict(lambda: 1.0)
    if args.recon_enabled:
        losses.append("recon")
        loss_weights["recon"] = args.recon_weight
    if args.sub_contra_s_enabled:
        losses.append("sub_contra_s")
        loss_weights["sub_contra_s"] = args.sub_contra_s_weight
    if args.task_contra_t_enabled:
        losses.append("task_contra_t")
        loss_weights["task_contra_t"] = args.task_contra_t_weight
    if args.latent_permute_s_enabled:
        losses.append("latent_permute_s")
        loss_weights["latent_permute_s"] = args.latent_permute_s_weight
    if args.latent_permute_t_enabled:
        losses.append("latent_permute_t")
        loss_weights["latent_permute_t"] = args.latent_permute_t_weight
    if args.restored_permute_s_enabled:
        losses.append("restored_permute_s")
        loss_weights["restored_permute_s"] = args.restored_permute_s_weight
    if args.restored_permute_t_enabled:
        losses.append("restored_permute_t")
        loss_weights["restored_permute_t"] = args.restored_permute_t_weight
    if args.sub_content_enabled:
        losses.append("sub_content")
        loss_weights["sub_content"] = args.sub_content_weight
    if args.task_content_enabled:
        losses.append("task_content")
        loss_weights["task_content"] = args.task_content_weight
    if args.sub_cross_s_enabled:
        losses.append("sub_cross_s")
        loss_weights["sub_cross_s"] = args.sub_cross_s_weight
    if args.task_cross_t_enabled:
        losses.append("task_cross_t")
        loss_weights["task_cross_t"] = args.task_cross_t_weight
    if args.scramble_permute_enabled:
        losses.append("scramble_permute")
        loss_weights["scramble_permute"] = args.scramble_permute_weight
    if args.conversion_permute_enabled:
        losses.append("conversion_permute")
        loss_weights["conversion_permute"] = args.conversion_permute_weight
    if args.quadruplet_permute_enabled:
        if args.quadruplet_permute_F_enabled:
            losses.append("quadruplet_permute_F")
            loss_weights["quadruplet_permute_F"] = args.quadruplet_permute_weight
        else:
            losses.append("quadruplet_permute")
            loss_weights["quadruplet_permute"] = args.quadruplet_permute_weight
    
    data_dir = os.path.join(args.root_dir, "dataset")
    csv_file = os.path.join(args.root_dir, args.csv_file)
    data_dict = pd.read_csv(csv_file)

    KFOLD = 5
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    unique_subjects = list(set(data_dict["subjects"]))

    for i, (train_idx, test_idx) in enumerate(kf.split(unique_subjects)):
        if i==1:
            break
        # Load data
        with torch.no_grad():
            #data_dict["data"].sub_(mean).div_(std)
            train_subjects = np.array(unique_subjects)[train_idx]
            test_subjects = np.array(unique_subjects)[test_idx]
            loader = SleepCustomLoader(data_dict, data_dir, subjects = train_subjects, split = 'train')
            test_loader = SleepCustomLoader(data_dict, data_dir, subjects = test_subjects, split = 'test')

            print("Data loaded")

        model = SplitLatentModel(IN_CHANNELS, args.channels, args.latent_dim, NUM_LAYERS, KERNEL_SIZE, recon_type=args.recon_type, content_cosine=args.content_cosine, time_resolution=200)
        model.set_losses(
            batch_size=args.batch_size,
            losses=losses,
            loader=loader,
            loss_weights=loss_weights,
        )
        numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {numel}", file=sys.stdout, flush=True)
    
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
        with torch.no_grad():
            model.losses()

        EFFECTIVE_BATCH_SIZE = loader.total_samples
        print(f"Effective batch size: {EFFECTIVE_BATCH_SIZE}", file=sys.stdout, flush=True)
        
        loss_notation = [n for l in model.used_losses for n in loss_to_notation[l.lower()]]
        loss_notation = sorted(set(loss_notation), key=loss_notation.index)
        loss_tags = "_".join(loss_notation).replace(":", "").replace("CRs_CRt", "CR").replace("LPs_LPt", "LP").replace("RPs_RPt", "RP").replace("Cs_Ct", "C").replace("CEs_CEt", "CE")
        print("Loss tags:", loss_tags, file=sys.stdout, flush=True)
        print("Used losses:", model.used_losses, file=sys.stdout, flush=True)
        
        BATCHES = (args.epochs * loader.size) // EFFECTIVE_BATCH_SIZE
        
        div_factor = args.max_lr / args.initial_lr

        if OLD_SCHED:
            scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            div_factor=div_factor,
            max_lr=args.max_lr,
            steps_per_epoch=1,
            epochs=BATCHES,
            three_phase=False,
            pct_start=args.pct_start,
            final_div_factor=args.final_div_factor,
            )
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=args.sched_mode,
                factor=args.sched_factor,
                patience=args.sched_patience,
            )

        config = {
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "max_lr": args.max_lr,
            "initial_lr": args.initial_lr,
            "pct_start": args.pct_start,
            "final_div_factor": args.final_div_factor,
            "root_dir": args.root_dir,
            "batches": BATCHES,
            "losses": model.used_losses,
            "seed": SEED,
            "in_channels": IN_CHANNELS,
            "channels": args.channels,
            "latent_dim": args.latent_dim,
            "num_layers": NUM_LAYERS,
            "num_params": numel,
            "effective_latent_dim": model.effective_latent_dim,
            "latent_seqs": model.latent_seqs,
            "recon_type": model.recon_type,
            "loss_tags": loss_tags,
            "eval_every": args.eval_every,
            "conversion_N": args.conversion_N,
        }
    
        for l in model.used_losses:
            config[f"{l}_weight"] = model.loss_weights[l]
        for l in all_losses:
            config[f"{l}_enabled"] = l in model.used_losses
        extra_tags = []
        if len(args.extra_tags) > 0:
            extra_tags = args.extra_tags.split(",")
        group = None
        if len(args.group) > 0:
            group = args.group
        data = {
            'config': config,
            'test_results': 0,
            'mse_results': 0,
            'test_knn_results': 0,
            'test_etc_results': 0,
        }


        # Start training
        # Creates a progress bar that goes from 0 to BATCHES-1. 
        # BATCHES represents the number of iterations the loop will perform.
        # EFFECTIVE_BATCH_SIZE represents the number of samples in each batch.

        #data['eval_results'] = []
        loss_list = defaultdict(list)
        with tqdm(range(BATCHES), unit_scale=EFFECTIVE_BATCH_SIZE, disable=not USE_TQDM, file=sys.stdout) as pbar:
            for i in pbar:
                model.train()
                optimizer.zero_grad()
                x, loss_dict = model.losses()
                total_loss = sum((model.loss_weights[v] * loss_dict[v] for v in model.used_losses))
                total_loss.backward()
                optimizer.step()
                if OLD_SCHED:
                    scheduler.step()

        # Evaluation
        model.loader = test_loader
        model.eval()
        print('Evaluating...', file=sys.stdout, flush=True)
        subject_latents, task_latents, subjects, tasks, runs, losses = get_split_latents(model, test_loader, test_loader.get_dataloader(batch_size=model.batch_size, random_sample=False))
        test_results = get_results(subject_latents, task_latents, subjects, tasks, split=test_loader.split, off_class_accuracy=args.full_eval)
        data['test_results'] = test_results
        if args.conversion_results:
            mse_results = get_full_conversion_results(model, test_loader, subject_latents, task_latents, args.conversion_N)
            data['mse_results'] += mse_results
        if args.extra_classifiers:
            test_knn_results += get_results(subject_latents, task_latents, subjects, tasks, clf='KNN', fit_clf=fit_knn_fn, split=test_loader.split)
            data['test_knn_results'] = test_knn_results
            test_etc_results += get_results(subject_latents, task_latents, subjects, tasks, clf='ETC', fit_clf=fit_etc_fn, split=test_loader.split)
            data['test_etc_results'] = test_etc_results

    print(data)
    '''
    identifier = np.random.randint(0, 10000)
    if args.save_model:
        save_as = args.model_save_dir+f"{identifier:04d}-{loss_tags}-{args.add_name}.pt"
        torch.save(model.state_dict(), save_as)
        print(f"Saved model to {save_as}", file=sys.stdout, flush=True)
    save_results_as = args.results_save_dir+f'{identifier:04d}-{loss_tags}-{args.add_name}.ckpt'
    torch.save(data, save_results_as)'''