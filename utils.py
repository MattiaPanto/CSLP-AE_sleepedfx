from collections import defaultdict
import numpy as np
from torch.utils.data import IterableDataset
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
from collections import Counter, defaultdict
import pandas as pd
import os
from collections import OrderedDict

def do_tsne(latents):
    tsne_ = TSNE(n_components=2, perplexity=30, n_jobs=-1, random_state=1968125571)
    tsne_latents = tsne_.fit_transform(latents)
    return tsne_latents, tsne_latents

def split_do_tsne(subject_latents, task_latents):
    return do_tsne(subject_latents)[0], do_tsne(task_latents)[0]


def plot_latents(fig, ax, latents, loader, which='subject', size=1, cmap=None):
    task_to_label = {
            1: "Sleep stage W",
            2: "Sleep stage 1",
            3: "Sleep stage 2",
            4: "Sleep stage 3/4",
            5: "Sleep stage R"
        }
    which_values = None
    which_values = loader.tasks if which == 'task' else loader.subjects
    which_uniques = loader.unique_tasks if which == 'task' else loader.unique_subjects
    if cmap is None:
        cmap = 'tab20' if which == 'task' else 'nipy_spectral'
    color_palette = sns.color_palette(cmap, len(which_uniques))
    idx_trans = {idx: i for i, idx in enumerate(which_uniques)}
    cluster_colors = [color_palette[x] for x in np.vectorize(idx_trans.get)(which_values)]
    cluster_colors = np.array(cluster_colors)
    for i in which_uniques:
        x, y = latents[which_values == i, 0], latents[which_values == i, 1]
        c = cluster_colors[which_values == i]
        ax.scatter(x, y, color=c[0], s=size if loader.split == 'train' else 2*size, label=f"{task_to_label[i]}" if which == 'task' else f"S{i}", alpha=0.5 if loader.split == 'train' else 1.0)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])


@torch.inference_mode(True)
def get_split_latents(model, loader, dataloader):
    model.eval()
    subject_latents = []
    task_latents = []
    subjects = []
    tasks = []
    runs = []
    losses = defaultdict(list)
    
    for idxs, xs, ss, ts, rs in dataloader:
        xs = xs.cuda()
        ss = ss.cuda()
        ts = ts.cuda()
        x_dict = {'x': xs, 'S': ss-1, 'T': ts}
        x_dict, loss_dict = model.losses(x_dict, model.used_losses, loader=loader)
        for l_name, l_val in loss_dict.items():
            losses[l_name].append(l_val)
        x_dict = model.get_s_t(x_dict)
        subject_latents.append(x_dict['s'])
        task_latents.append(x_dict['t'])
        subjects.extend(ss.tolist())
        tasks.extend(ts.tolist())
        runs.extend(rs.tolist())
    subject_latents = torch.cat(subject_latents, dim=0).cpu().numpy()
    task_latents = torch.cat(task_latents, dim=0).cpu().numpy()
    subjects = np.array(subjects)
    tasks = np.array(tasks)
    runs = np.array(runs)
    for l_name, l_val in losses.items():
        losses[l_name] = torch.stack([l.mean() for l in l_val], dim=0).mean().item()
    return subject_latents, task_latents, subjects, tasks, runs, losses


def fit_etc_fn(X, y):
    clf = ExtraTreesClassifier(n_estimators=200, max_features=0.2, max_depth=60, min_samples_split=2, bootstrap=False, class_weight="balanced", n_jobs=-1, random_state=1968125571)
    #clf = ExtraTreesClassifier(n_estimators=250, max_depth=20, min_samples_split=150, bootstrap=False, class_weight="balanced", n_jobs=-1, random_state=1968125571)
    clf.fit(X, y)
    return clf

def fit_knn_fn(X, y):
    clf = KNeighborsClassifier(n_neighbors=15, n_jobs=-1, metric='cosine')
    clf.fit(X, y)
    return clf


class XGBWrapper:
    
    def __init__(self, xgb_object):
        self.xgb_object = xgb_object
    
    def fit(self, X, y):
        self.translation_dict = {l: i for i, l in enumerate(np.unique(y))}
        self.retranslation_dict = {i: l for i, l in enumerate(np.unique(y))}
        y = np.vectorize(self.translation_dict.get)(y)
        class_counts = Counter(y)
        class_weights = {i: min(class_counts.values()) / class_counts[i] for i in class_counts.keys()}
        class_weights_arr = np.vectorize(class_weights.get)(y)
        self.xgb_object.fit(X, y, sample_weight=class_weights_arr)
    
    def predict(self, X):
        y_pred = self.xgb_object.predict(X)
        y_pred = np.vectorize(self.retranslation_dict.get)(y_pred)
        return y_pred
    
    def score(self, X, y):
        y = np.vectorize(self.translation_dict.get)(y)
        score = self.xgb_object.score(X, y)
        return score


def fit_clf_fn(X, y):
    clf = XGBWrapper(xgb.XGBClassifier(n_estimators=300, max_bin=100, learning_rate=0.3, grow_policy='depthwise', objective='multi:softmax', tree_method='gpu_hist', n_jobs=-1))
    clf.fit(X, y)
    return clf

def balanced_sample(which, sampling_method='under'):
    # Creates a balanced sample by either undersampling or oversampling classes.

    #Prepare for undersampling
    which_idxs = defaultdict(list)
    for i, t in enumerate(which):
        which_idxs[t].append(i)
    count_fn = min if sampling_method == 'under' else max
    count = count_fn([len(idxs) for idxs in which_idxs.values()])
    #Create flat index for #count samples from each task
    flat_idxs = []
    for t, idxs in which_idxs.items():
        #random sample #count samples from task t
        flat_idxs.extend(np.random.choice(idxs, size=count, replace=not sampling_method.lower().startswith('u')))
    return flat_idxs

def extract_block_diag(a, n, k=0):
    #credit: pv. from https://stackoverflow.com/a/10862636 12/04/2023
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("Only 2-D arrays handled")
    if not (n > 0):
        raise ValueError("Must have n >= 0")

    if k > 0:
        a = a[:,n*k:] 
    else:
        a = a[-n*k:]

    n_blocks = min(a.shape[0]//n, a.shape[1]//n)

    new_shape = (n_blocks, n, n)
    new_strides = (n*a.strides[0] + n*a.strides[1],
                   a.strides[0], a.strides[1])

    return np.lib.stride_tricks.as_strided(a, new_shape, new_strides)

def get_results(subject_latents, task_latents, subjects, tasks, split='test', clf='XGB', fit_clf=fit_clf_fn, sampling_method='under', off_class_accuracy=False):
    test_results = {}
    
    #Create the cross validation splits
    #NEW
    '''
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1968125571)
    '''
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1968125571)
    
    #Create the subject and task splits
    #subject_splits = cv.split(subject_latents, subjects, groups=tasks)
    #task_splits = cv.split(task_latents, tasks, groups=subjects)
    subject_splits = cv.split(subject_latents, subjects)
    task_splits = cv.split(task_latents, tasks)
    
    subject_scores = []
    task_scores = []
    
    subject_cms = []
    task_cms = []
    
    task_on_subject_scores = []
    subject_on_task_scores = []
    
    subject_metrics = defaultdict(list)
    
    task_metrics = defaultdict(list)
    
    for i, (train_idx, test_idx) in enumerate(tqdm(subject_splits)):
        #Get the training and testing data
        X_train, X_test = subject_latents[train_idx], subject_latents[test_idx]
        subjects_train, subjects_test = subjects[train_idx], subjects[test_idx]
        tasks_train, tasks_test = tasks[train_idx], tasks[test_idx]

        flat_idxs = balanced_sample(subjects_train, sampling_method=sampling_method)
        X_train = X_train[flat_idxs]
        subjects_train = subjects_train[flat_idxs]
        tasks_train = tasks_train[flat_idxs]
        
        clf_subject = fit_clf(X_train, subjects_train)
        subject_scores.append(clf_subject.score(X_test, subjects_test))
        
        subject_cm = confusion_matrix(subjects_test, clf_subject.predict(X_test))
        subject_cms.append(subject_cm)
        
        y_true = subjects_test
        y_pred = clf_subject.predict(X_test)
        
        subject_metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        subject_metrics['precision'].append(precision_score(y_true, y_pred, average='weighted'))
        subject_metrics['recall'].append(recall_score(y_true, y_pred, average='weighted'))
        subject_metrics['f1'].append(f1_score(y_true, y_pred, average='weighted'))
        subject_metrics['balanced_accuracy'].append(balanced_accuracy_score(y_true, y_pred))
        
        precision_scores = precision_score(y_true, y_pred, average=None, labels=np.unique(y_true))
        recall_scores = recall_score(y_true, y_pred, average=None, labels=np.unique(y_true))
        f1_scores = f1_score(y_true, y_pred, average=None, labels=np.unique(y_true))
        for s, pr, re, f1 in zip(np.unique(y_true), precision_scores, recall_scores, f1_scores):
            subject_metrics['precision_s' + str(s)].append(pr)
            subject_metrics['recall_s' + str(s)].append(re)
            subject_metrics['f1_s' + str(s)].append(f1)
            
        if off_class_accuracy:
            X_task_train, X_task_test = task_latents[train_idx][flat_idxs], task_latents[test_idx]
            clf_subject = fit_clf(X_task_train, subjects_train)
            y_true = subjects_test
            y_pred = clf_subject.predict(X_task_test)
            subject_on_task_scores.append(balanced_accuracy_score(y_true, y_pred))
  
    
    for i, (train_idx, test_idx) in enumerate(tqdm(task_splits)):
        #Get the training and testing data
        X_train, X_test = task_latents[train_idx], task_latents[test_idx]
        subjects_train, subjects_test = subjects[train_idx], subjects[test_idx]
        tasks_train, tasks_test = tasks[train_idx], tasks[test_idx]

        flat_idxs = balanced_sample(tasks_train, sampling_method=sampling_method)
        X_train = X_train[flat_idxs]
        subjects_train = subjects_train[flat_idxs]
        tasks_train = tasks_train[flat_idxs]
        
        clf_task = fit_clf(X_train, tasks_train)
        task_scores.append(clf_task.score(X_test, tasks_test))
        
        task_cm = confusion_matrix(tasks_test, clf_task.predict(X_test))
        task_cms.append(task_cm)
        
        y_true = tasks_test
        y_pred = clf_task.predict(X_test)
        
        task_metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        task_metrics['precision'].append(precision_score(y_true, y_pred, average='weighted'))
        task_metrics['recall'].append(recall_score(y_true, y_pred, average='weighted'))
        task_metrics['f1'].append(f1_score(y_true, y_pred, average='weighted'))
        task_metrics['balanced_accuracy'].append(balanced_accuracy_score(y_true, y_pred))
        
        precision_scores = precision_score(y_true, y_pred, average=None, labels=np.unique(y_true))
        recall_scores = recall_score(y_true, y_pred, average=None, labels=np.unique(y_true))
        f1_scores = f1_score(y_true, y_pred, average=None, labels=np.unique(y_true))

        for s, pr, re, f1 in zip(np.unique(y_true), precision_scores, recall_scores, f1_scores):
            task_metrics['precision_t' + str(s)].append(pr)
            task_metrics['recall_t' + str(s)].append(re)
            task_metrics['f1_t' + str(s)].append(f1)
        
        if off_class_accuracy:
            X_subject_train, X_subject_test = subject_latents[train_idx][flat_idxs], subject_latents[test_idx]
            clf_task = fit_clf(X_subject_train, tasks_train)
            y_true = tasks_test
            y_pred = clf_task.predict(X_subject_test)
            task_on_subject_scores.append(balanced_accuracy_score(y_true, y_pred))
 
    #Summarize the results
    subject_score = np.mean(subject_scores)
    task_score = np.mean(task_scores)
    

    test_results['clf'] = clf
    
    test_results[clf + '/' + split + '/subject/score'] = subject_score
    test_results[clf + '/' + split + '/task/score'] = task_score
    test_results[clf + '/' + split + '/subject/scores'] = subject_scores
    test_results[clf + '/' + split + '/task/scores'] = task_scores
    
    if off_class_accuracy:
        subject_on_task_score = np.mean(subject_on_task_scores)
        task_on_subject_score = np.mean(task_on_subject_scores)
        test_results[clf + '/' + split + '/task/subject_on_task_score'] = subject_on_task_score
        test_results[clf + '/' + split + '/subject/task_on_subject_score'] = task_on_subject_score
        test_results[clf + '/' + split + '/task/subject_on_task_scores'] = subject_on_task_scores
        test_results[clf + '/' + split + '/subject/task_on_subject_scores'] = task_on_subject_scores
    
    test_results[clf + '/' + split + '/subject/cm'] = np.sum(subject_cms, axis=0) # Confusion matrices for subjects
    test_results[clf + '/' + split + '/task/cm'] = np.sum(task_cms, axis=0) # Confusion matrices for tasks

    #task_cm_result = test_results[clf + '/' + split + '/task/cm']
    #task_cm_true = np.sum(task_cm_result, axis=1)
    #task_cm_correct = np.diag(task_cm_result)
    #task_class_accuracy = task_cm_correct / task_cm_true
    #test_results[clf + '/' + split + '/task/accuracy'] = np.mean(task_class_accuracy)



    for k, v in subject_metrics.items():
        test_results[clf + '/' + split + '/subject/' + k] = np.mean(v)
    for k, v in task_metrics.items():
        test_results[clf + '/' + split + '/task/' + k] = np.mean(v)
    return test_results

def get_eval_results(subject_latents, task_latents, subjects, tasks, split='eval', clf='XGB', fit_clf=fit_clf_fn, sampling_method='under'):
    #Same as above, but for a stratified split
    test_results = {}
    test_results['clf'] = clf

    # Tasks eval
    X_task_train, X_task_test, tasks_train, tasks_test = train_test_split(task_latents, tasks, stratify=tasks, test_size=0.2, shuffle=True, random_state=1968125571)
    flat_idxs = balanced_sample(tasks_train, sampling_method=sampling_method)
    X_task_train = X_task_train[flat_idxs]
    tasks_train = tasks_train[flat_idxs]

    task_clf = fit_clf(X_task_train, tasks_train)
    y_pred = task_clf.predict(X_task_test)
    y_true = tasks_test
    task_score = balanced_accuracy_score(y_true, y_pred)
    test_results[clf + '/' + split + '/task/score'] = task_score
    
    # Subjects eval
    X_subject_train, X_subject_test, subjects_train, subjects_test = train_test_split(subject_latents, subjects, stratify=subjects, test_size=0.2, shuffle=True, random_state=1968125571)
    flat_idxs = balanced_sample(subjects_train, sampling_method=sampling_method)
    X_subject_train = X_subject_train[flat_idxs]
    subjects_train = subjects_train[flat_idxs]

    subject_clf = fit_clf(X_subject_train, subjects_train)
    y_pred = subject_clf.predict(X_subject_test)
    y_true = subjects_test
    subject_score = balanced_accuracy_score(y_true, y_pred)
    test_results[clf + '/' + split + '/subject/score'] = subject_score

    #paradigm_trans = {i: i // 2 for i in np.unique(tasks_train)}
    #y_true_trans = np.vectorize(paradigm_trans.get)(y_true)
    #y_pred_trans = np.vectorize(paradigm_trans.get)(y_pred)
    #paradigm_score = balanced_accuracy_score(y_true_trans, y_pred_trans)
    #test_results[clf + '/' + split + '/task/paradigm_wise_accuracy'] = paradigm_score
    test_results[clf + '/' + split + '/mean_score'] = np.mean([subject_score, task_score]) 
    return test_results


    

class DelegatedLoader(IterableDataset):
    
    def __init__(self, loader, property=None, batch_size=None, length=None):
        self.loader = loader
        self._property = property
        self._batch_size = batch_size
        self._length = length #total_sample
    
    def __len__(self):
        if self._batch_size is not None or self._property is not None:
            if self._length is not None:
                if self._batch_size is not None:
                    return self._length // self._batch_size
                return self._length
            return None
        return self.size
    
    def __iter__(self):
        if self._batch_size is not None or self._property is not None:
            if self._batch_size is not None:
                return self.loader.batch_iterator(self._batch_size, self._length)
            elif self._property is not None:
                return self.loader.property_iterator(self._property, self._length)
        else:
            return self.loader.iterator()

class CustomLoader():
    
    def __init__(self, data_dict, split_subjects, split='train', location = None):
        self.data = data_dict["data"]
        
        device_type = self.data.device.type
        
        self.split = split

        def to_numpy(tensor):
            return tensor.cpu().numpy() if device_type == 'cuda' else tensor.numpy()

        self.subjects = to_numpy(data_dict["subjects"])
        self.tasks = to_numpy(data_dict["tasks"])
        self.runs = to_numpy(data_dict["runs"])

        self.data_mean = self.data.mean().detach().clone().contiguous().cuda()
        self.data_std = self.data.std().detach().clone().contiguous().cuda()
        
        self.task_to_label = {
            1: "Sleep stage W",
            2: "Sleep stage 1",
            3: "Sleep stage 2",
            4: "Sleep stage 3/4",
            5: "Sleep stage R"
        } #connect task ids to a string
        self.run_labels = ['First night', 'Second night']

        self.unique_subjects = split_subjects
        self.unique_tasks = list(self.task_to_label.keys())
        self.unique_runs = list(range(1, len(self.run_labels) + 1))

        self.subject_indices = {s: [] for s in self.unique_subjects} #A dict with a list for each subject in splits
        self.task_indices = {t: [] for t in self.unique_tasks}       #A dict with a list for each task in splits
        self.run_indices = {r: [] for r in self.unique_runs}         #A dict with a list for each run in splits
        
        # sample_indices will contain the ids of the samples that are present in the split, based on the selected subjects.
        self.total_samples = 0
        data_indices = []
        for i, (s, t, r) in enumerate(zip(self.subjects, self.tasks, self.runs)):
            if s not in self.subject_indices:
                continue
            if t not in self.task_indices:
                continue
            if r not in self.run_indices:
                continue
            data_indices.append(i)
        #self.data = self.data[data_indices].float().contiguous().detach().clone()
        if location == "cpu":
            self.data = self.data[data_indices].float().contiguous().detach().clone().cpu() # NEW
        elif location == "cuda":
            self.data = self.data[data_indices].float().contiguous().detach().clone().cuda() # NEW
        else:
            self.data = self.data[data_indices].float().contiguous().detach().clone() # NEW
        torch.cuda.empty_cache()
        
        # Convert to a contiguous array in memory
        self.subjects = np.ascontiguousarray(self.subjects[data_indices]) 
        self.tasks = np.ascontiguousarray(self.tasks[data_indices])
        self.runs = np.ascontiguousarray(self.runs[data_indices])
        self.size = len(self.data)

        self.full_indices = defaultdict(lambda: defaultdict(list))
        # For each subject, task, or run, the dictionary returns a list of sample ids
        for i, (s, t, r) in enumerate(zip(self.subjects, self.tasks, self.runs)):
            self.subject_indices[s].append(i)
            self.task_indices[t].append(i)
            self.run_indices[r].append(i)
            self.full_indices[s][t].append(i)

        

    def reset_sample_counts(self):
        self.total_samples = 0
    
    def get_dataloader(self, num_total_samples=None, batch_size=None, property=None, random_sample=True):
        delegated_loader = DelegatedLoader(self, property=property, batch_size=batch_size if random_sample else None, length=num_total_samples)
        if not random_sample and batch_size is not None:
            return DataLoader(delegated_loader, batch_size=batch_size, pin_memory=False)
        return DataLoader(delegated_loader, batch_size=None, pin_memory=False)
    
    def sample_by_condition(self, subjects, tasks):
        samples = []
        for s, t in zip(subjects, tasks):
            i = np.random.choice(self.full_indices[s][t])
            samples.append(i)
        samples = np.array(samples)
        self.total_samples += len(samples)

        return self.data[samples]
    
    def sample_by_property(self, property):
        
        property = property.lower()
        if property.startswith("s"):
            property_indices = self.subject_indices
        elif property.startswith("t"):
            property_indices = self.task_indices
        elif property.startswith("r"):
            property_indices = self.run_indices
        else:
            raise ValueError("Invalid property")
        
        samples = []
        for indices in property_indices.values():
            i = np.random.choice(indices)
            samples.append(i)
        samples = np.array(samples)
        self.total_samples += len(samples)

        return samples, self.data[samples], self.subjects[samples], self.tasks[samples], self.runs[samples]
    
    def sample_batch(self, batch_size):
        samples = np.random.randint(0, self.size, size=batch_size)
        self.total_samples += batch_size
        return samples, self.data[samples], self.subjects[samples], self.tasks[samples], self.runs[samples]
    
    def iterator(self):
        for i in range(self.size):
            self.total_samples += 1
            yield i, self.data[i], self.subjects[i], self.tasks[i], self.runs[i]
    
    def batch_iterator(self, batch_size, length):
        # Return batches of size batch_size, until length is reached.
        num_samples = 0
        while True:
            if length is not None and num_samples + batch_size >= length:
                break
            yield self.sample_batch(batch_size)
            num_samples += batch_size
    
    def property_iterator(self, property, length):
        # Return batches with unique property (subject/task/run), until length is reached.
        num_samples = 0
        num_per = 0
        while True:
            if length is not None and num_samples + num_per >= length:
                break
            yield self.sample_by_property(property)
            if length is not None:
                if num_per == 0:
                    property = property.lower()
                    if property.startswith("s"):
                        num_per = len(self.subject_indices)
                    elif property.startswith("t"):
                        num_per = len(self.task_indices)
                    elif property.startswith("r"):
                        num_per = len(self.run_indices)
                    else:
                        raise ValueError("Invalid property")
                num_samples += num_per