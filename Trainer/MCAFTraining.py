from pathlib import Path
import torch
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler
from tqdm import tqdm
from utils.metric import Metric, SubMetric
from utils.store import save_state
import re
from utils.experiment_logger import ExperimentLogger, get_current_lr

def train(model, dataset_pretrain, dataset_train, dataset_val, dataset_test, device, output_dir="result/", 
          metrics=None, metric_choose=None, optimizer=None, scheduler=None, 
          batch_size=32, epochs=50, criterion=None,loss_func=None, loss_param= None,test_sub_label = None,
          log_context=None):
    if metrics is None:
        metrics=['acc']
    if metric_choose is None:
        metric_choose = metrics[0]

    loss_metics = ['loss']
    
    #data sampler for train, val, test
    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    sampler_test = SequentialSampler(dataset_test)

    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4)
    data_loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=batch_size, num_workers=4)
    data_loader_test = DataLoader(dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4)
    test_sub_label_loader = DataLoader(
        test_sub_label, sampler=sampler_test, batch_size=batch_size, num_workers=4, drop_last=False
    ) if test_sub_label is not None else None
    model = model.to(device)
    logger = ExperimentLogger(output_dir, log_context)
    best_metric = {m: 0.0 for m in metrics}
    
    for epoch in range(epochs):
        model.train()
        metric = Metric(metrics)
        train_bar = tqdm(enumerate(data_loader_train), total=len(data_loader_train),
                         desc=f"Train Epoch {epoch+1}/{epochs}: lr={optimizer.param_groups[0]['lr']:.6f}")
        
        for idx, (eeg_data, eog_data, targets) in train_bar:
            eeg_data = eeg_data.to(device)
            eog_data = eog_data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(eeg_data, eog_data)
            loss = criterion(outputs, targets) + 0.01 * sum(p.norm() for p in model.parameters())  # L2正则化
            metric.update(torch.argmax(outputs, dim=1), targets, loss.item())
            train_bar.set_postfix_str(f"loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        train_metrics = metric.to_dict()
        metric_value = evaluate(model, data_loader_val, device, metrics, criterion)
        improved_metrics = []
        for m in metrics:
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                improved_metrics.append(m)
                save_state(output_dir, model, optimizer,epoch, metric=m, state='best')
        logger.log_epoch(epoch + 1, epochs, get_current_lr(optimizer), train_metrics, metric_value, best_metric, improved_metrics)
    
    model.load_state_dict(torch.load(f'{output_dir}/checkpoint-best{metric_choose}')['model'])
    if test_sub_label is not None:
        metric_value = sub_evaluate(model, data_loader_test, test_sub_label_loader, device, metrics, criterion, loss_func, loss_param)
    else:
        metric_value = evaluate(model, data_loader_test, device, metrics, criterion, loss_func, loss_param)
    logger.log_final(metric_value, best_metric, metrics)

    return metric_value

@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion, loss_func=None, loss_param=None):
    model.eval()
    metric = Metric(metrics)
    
    for idx, (eeg_data, eog_data, targets) in tqdm(enumerate(data_loader), 
                                                  total=len(data_loader), desc="Evaluating"):
        eeg_data = eeg_data.to(device)
        eog_data = eog_data.to(device)
        targets = targets.to(device)
        
        outputs = model(eeg_data, eog_data)
        loss = criterion(outputs, targets) + 0.01 * sum(p.norm() for p in model.parameters())
        
        metric.update(torch.argmax(outputs, dim=1), targets, loss.item())
    
    return metric.to_dict()

@torch.no_grad()
def sub_evaluate(model, data_loader, sub_labels, device, metrics, criterion, loss_func, loss_param):
    model.eval()
    metric = SubMetric(metrics)
    
    eval_bar = tqdm(enumerate(zip(data_loader, sub_labels)), total= len(data_loader), desc = 'Evaluating:')

    for idx,((eeg_features, bio_features, labels), sub_label) in eval_bar:
        eeg_features = eeg_features.to(device)
        bio_features = bio_features.to(device)
        labels = labels.to(device)
        
        prediction = model(eeg_features, bio_features)
        loss = criterion(prediction, labels)
        
        metric.update(torch.argmax(prediction, dim=1), labels, sub_label, loss.item())
    
    return metric.to_dict()
