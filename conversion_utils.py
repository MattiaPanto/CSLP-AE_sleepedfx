
import numpy as np
import torch
from torch.nn import functional as F
from collections import defaultdict

@torch.inference_mode(True)
def sample_latents(loader, subject_latents, task_latents, target_subject, target_task, n=2000, specific_task='all', specific_subject='all'):

    # Sample a subset of n subject and task latents given a target.
    # es:
    # subject 0, task 0
    # subject 0, task 1
    # subject 1, task 0
    # subject 1, task 1
    # subject 2, task 0
    # subject 2, task 1

    pass
    if isinstance(specific_subject, int):
        convert_task_latents = task_latents[(loader.subjects == specific_subject) & (loader.tasks == target_task)]
    elif specific_subject == 'all':
        convert_task_latents = task_latents[loader.tasks == target_task]
    elif specific_subject == 'same':
        convert_task_latents = task_latents[(loader.subjects == target_subject) & (loader.tasks == target_task)]
    elif specific_subject == 'different':
        convert_task_latents = task_latents[(loader.subjects != target_subject) & (loader.tasks == target_task)]
    else:
        raise ValueError('specific_subject must be one of [#subject_class_label#, all, same, different]')
    
    if isinstance(specific_task, int):
        convert_subject_latents = subject_latents[(loader.subjects == target_subject) & (loader.tasks == specific_task)]
    elif specific_task == 'all':
        convert_subject_latents = subject_latents[loader.subjects == target_subject]
    elif specific_task == 'same':
        convert_subject_latents = subject_latents[(loader.subjects == target_subject) & (loader.tasks == target_task)]
    elif specific_task == 'different':
        convert_subject_latents = subject_latents[(loader.subjects == target_subject) & (loader.tasks != target_task)]
    else:
        raise ValueError('specific_task must be one of [#task_class_label#, all, same, different]')
    

    if convert_task_latents.shape[0] == 0:
        raise ValueError(f'No samples found for subject {target_subject} and task {target_task}')
    
    # If there are less than n latents, sample with replacement
    # If there are more than n latents, sample without replacement
    num_task_latents = convert_task_latents.shape[0]
    if num_task_latents < n:
        task_permute_idxs = np.random.randint(0, num_task_latents, size=n)
    else:
        task_permute_idxs = np.random.permutation(num_task_latents)[:n]
    convert_task_latents = convert_task_latents[task_permute_idxs]
    
    num_subject_latents = convert_subject_latents.shape[0]
    if num_subject_latents < n:
        subject_permute_idxs = np.random.randint(0, num_subject_latents, size=n)
    else:
        subject_permute_idxs = np.random.permutation(num_subject_latents)[:n]
    convert_subject_latents = convert_subject_latents[subject_permute_idxs]
    
    return convert_subject_latents, convert_task_latents


@torch.inference_mode(True)
def reconstruct(model, convert_subject_latents, convert_task_latents, batch_size=2048):
    num_latents = convert_subject_latents.shape[0]
    num_batches = int(np.ceil(num_latents / batch_size))
    convert_subject_latents = torch.unflatten(torch.tensor(convert_subject_latents, device='cuda'), 1, (model.latent_dim, model.latent_seqs))
    convert_task_latents = torch.unflatten(torch.tensor(convert_task_latents, device='cuda'), 1, (model.latent_dim, model.latent_seqs))
    reconstructions = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i+1) * batch_size
        x_dict = {'s': convert_subject_latents[start_idx:end_idx], 't': convert_task_latents[start_idx:end_idx]}
        x_dict = model.get_x_hat(x_dict)
        reconstructions.append(x_dict['x_hat'])
    return torch.cat(reconstructions, dim=0)


@torch.inference_mode(True)
def get_reconstructed_stft(model, loader, subject_latents, task_latents, t_spec, s_spec, target_subject, target_task, n):
    convert_subject_latents, convert_task_latents = sample_latents(
        loader, subject_latents, task_latents, 
        target_subject, target_task, 
        n=n, specific_task=t_spec, specific_subject=s_spec
    )
    reconstructions = reconstruct(model, convert_subject_latents, convert_task_latents)
    reconstructions = reconstructions * loader.data_std + loader.data_mean
    reconstructed_stft = reconstructions.mean(0)

    return reconstructed_stft

    


@torch.inference_mode(True)
def get_conversion_results(model, loader, subject_latents, task_latents, target_subject, target_task, channel, n):
    # Prende tutti i sample che corrispondono al soggetto e task1 target e calcola la media.
    real_stft = loader.data[(loader.subjects == target_subject) & (loader.tasks == target_task)]
    real_stft = real_stft.cuda() * loader.data_std + loader.data_mean
    real_stft = real_stft.mean(0)
    
    # Get the reconstructed STFT
    recon_stft_ss = get_reconstructed_stft(model, loader, subject_latents, task_latents, 'same', 'same', target_subject, target_task, n)
    recon_stft_sd = get_reconstructed_stft(model, loader, subject_latents, task_latents, 'same', 'different', target_subject, target_task, n)
    recon_stft_ds = get_reconstructed_stft(model, loader, subject_latents, task_latents, 'different', 'same', target_subject, target_task, n)
    recon_stft_dd = get_reconstructed_stft(model, loader, subject_latents, task_latents, 'different', 'different', target_subject, target_task, n)

    ss_mse = F.mse_loss(recon_stft_ss, real_stft)
    sd_mse = F.mse_loss(recon_stft_sd, real_stft)
    ds_mse = F.mse_loss(recon_stft_ds, real_stft)
    dd_mse = F.mse_loss(recon_stft_dd, real_stft)

    '''
    ss_mse = F.mse_loss(recon_stft_ss[channel], real_stft[channel])
    sd_mse = F.mse_loss(recon_stft_sd[channel], real_stft[channel])
    ds_mse = F.mse_loss(recon_stft_ds[channel], real_stft[channel])
    dd_mse = F.mse_loss(recon_stft_dd[channel], real_stft[channel])
    
    # Get the reconstructed STFT
    recon_stft_ss1, recon_stft_ss2 = get_reconstructed_stft(model, loader, subject_latents, task_latents, 'same', 'same', target_subject, target_task1, target_task2, n)
    recon_stft_sd1, recon_stft_sd2 = get_reconstructed_stft(model, loader, subject_latents, task_latents, 'same', 'different', target_subject, target_task1, target_task2, n)
    recon_stft_ds1, recon_stft_ds2 = get_reconstructed_stft(model, loader, subject_latents, task_latents, 'different', 'same', target_subject, target_task1, target_task2, n)
    recon_stft_dd1, recon_stft_dd2 = get_reconstructed_stft(model, loader, subject_latents, task_latents, 'different', 'different', target_subject, target_task1, target_task2, n)

    ss_mse = F.mse_loss(recon_stft_ss1[channel], real_stft1[channel]) + F.mse_loss(recon_stft_ss2[channel], real_stft2[channel])
    sd_mse = F.mse_loss(recon_stft_sd1[channel], real_stft1[channel]) + F.mse_loss(recon_stft_sd2[channel], real_stft2[channel])
    ds_mse = F.mse_loss(recon_stft_ds1[channel], real_stft1[channel]) + F.mse_loss(recon_stft_ds2[channel], real_stft2[channel])
    dd_mse = F.mse_loss(recon_stft_dd1[channel], real_stft1[channel]) + F.mse_loss(recon_stft_dd2[channel], real_stft2[channel])
    '''
    return ss_mse, sd_mse, ds_mse, dd_mse

@torch.inference_mode(True)
def get_full_conversion_results(model, test_loader, subject_latents, task_latents, N):
    ss_mses, sd_mses, ds_mses, dd_mses = [], [], [], []
    mse_results = {}

    # subj [0,1,2,3,4]
    # task [0,1,2]
    # target = (0,1)
    subject_task_pairs = set(zip(test_loader.subjects, test_loader.tasks))

    for target_subject in test_loader.unique_subjects:
        for target_task in test_loader.unique_tasks:  
            if (target_subject, target_task) not in subject_task_pairs:
                continue
            #channel = channel_to_idx[task_to_channel[target_task1]]
            channel = 0
            ss_mse, sd_mse, ds_mse, dd_mse = get_conversion_results(model, test_loader, subject_latents, task_latents, target_subject, target_task, channel, N)
            task_name = test_loader.task_to_label[target_task]
            mse_results[f'MSE/test/{task_name}/{target_subject}/ss'] = ss_mse
            mse_results[f'MSE/test/{task_name}/{target_subject}/sd'] = sd_mse
            mse_results[f'MSE/test/{task_name}/{target_subject}/ds'] = ds_mse
            mse_results[f'MSE/test/{task_name}/{target_subject}/dd'] = dd_mse
            ss_mses.append(ss_mse)
            sd_mses.append(sd_mse)
            ds_mses.append(ds_mse)
            dd_mses.append(dd_mse)

    #Calculate per subject mean
    for target_subject in test_loader.unique_subjects:
        ss_results, sd_results, ds_results, dd_results = [], [], [], []

        for target_task in test_loader.unique_tasks:
            if (target_subject, target_task) not in subject_task_pairs:
                continue
            task_name = test_loader.task_to_label[target_task]
            ss_results.append(mse_results[f'MSE/test/{task_name}/{target_subject}/ss'])
            sd_results.append(mse_results[f'MSE/test/{task_name}/{target_subject}/sd'])
            ds_results.append(mse_results[f'MSE/test/{task_name}/{target_subject}/ds'])
            dd_results.append(mse_results[f'MSE/test/{task_name}/{target_subject}/dd'])
        mse_results[f'MSE/test/mean/{target_subject}/ss'] = torch.mean(torch.stack(ss_results))
        mse_results[f'MSE/test/mean/{target_subject}/sd'] = torch.mean(torch.stack(sd_results))
        mse_results[f'MSE/test/mean/{target_subject}/ds'] = torch.mean(torch.stack(ds_results))
        mse_results[f'MSE/test/mean/{target_subject}/dd'] = torch.mean(torch.stack(dd_results))
    #Calculate per paradigm mean
    for target_task in test_loader.unique_tasks:
        task_name = test_loader.task_to_label[target_task]
        ss_results, sd_results, ds_results, dd_results = [], [], [], []

        for target_subject in test_loader.unique_subjects:
            if (target_subject, target_task) not in subject_task_pairs:
                continue
            ss_results.append(mse_results[f'MSE/test/{task_name}/{target_subject}/ss'])
            sd_results.append(mse_results[f'MSE/test/{task_name}/{target_subject}/sd'])
            ds_results.append(mse_results[f'MSE/test/{task_name}/{target_subject}/ds'])
            dd_results.append(mse_results[f'MSE/test/{task_name}/{target_subject}/dd'])
        mse_results[f'MSE/test/mean/{task_name}/ss'] = torch.mean(torch.stack(ss_results))
        mse_results[f'MSE/test/mean/{task_name}/sd'] = torch.mean(torch.stack(sd_results))
        mse_results[f'MSE/test/mean/{task_name}/ds'] = torch.mean(torch.stack(ds_results))
        mse_results[f'MSE/test/mean/{task_name}/dd'] = torch.mean(torch.stack(dd_results))
    
    #Calculate overall mean
    mse_results['MSE/test/mean/ss'] = torch.mean(torch.stack(ss_mses))
    mse_results['MSE/test/mean/sd'] = torch.mean(torch.stack(sd_mses))
    mse_results['MSE/test/mean/ds'] = torch.mean(torch.stack(ds_mses))
    mse_results['MSE/test/mean/dd'] = torch.mean(torch.stack(dd_mses))
    return mse_results
