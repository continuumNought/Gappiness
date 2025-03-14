import numpy as np
import optuna
import torch
import torch.nn as nn
import time
import functools as ft
from torch.utils.tensorboard import SummaryWriter
from gappiness.loss import FastJacobianRegularizedLoss, MSELossWrapper
from gappiness.model import Autoencoder
from gappiness.data import standard_normalize, load_shuffle, normalize_split_batch
from gappiness.noise import add_gaussian_noise
from multiprocessing import Process
from optuna.samplers import RandomSampler


class NodeVolumeLimitSampler(RandomSampler):
    def __init__(self, volume, base_sampler=None):
        super().__init__()
        self.base_sampler = base_sampler or RandomSampler()
        self.volume = volume

    def sample_independent(self, study, trial, param_name, param_distribution):
        while True:
            value = self.base_sampler.sample_independent(study, trial, param_name, param_distribution)

            hidden_dim = trial.params.get("hidden_dim", value if param_name == "hidden_dim" else None)
            encoded_dim = trial.params.get("encoded_dim", value if param_name == "encoded_dim" else None)
            hidden_layers = trial.params.get("hidden_layers", value if param_name == "hidden_layers" else None)

            if hidden_dim is not None and encoded_dim is not None \
                and hidden_dim < encoded_dim:
                continue

            if hidden_dim is not None and hidden_layers is not None \
                and hidden_dim*hidden_layers > self.volume:
                continue

            return value


def select_loss(loss, model, sigma_sqr):
    if loss == 'FJL':
        return FastJacobianRegularizedLoss(model, sigma_sqr=sigma_sqr)

    return MSELossWrapper()


class EarlyStop:
    def __init__(self, improve_ratio=0.9, patience=5, warmup=10):
        self.patience = patience
        self.warmup = warmup
        self.misses = 0
        self.last_loss = None
        self.improve_ratio=improve_ratio
        self.epoch = 0
        assert(warmup >= patience)

    def stop(self, loss):
        self.epoch += 1
        if self.epoch <= self.warmup:
            self.last_loss = loss
            return False

        ratio_change = loss/self.last_loss
        self.last_loss = loss
        if ratio_change <= self.improve_ratio:
            self.misses = 0

        else:
            self.misses += 1

        return self.misses >= self.patience


def objective_factory(data_path, input_dim, max_epochs=100, normalize=None, **kwargs):
    hyperparam_sample_args = {
        'batch_sz': {'args': ('batch_sz', [16, 32, 64]), 'kwargs': {}},
        'hidden_dim': {'args': ('hidden_dim', 5, 205), 'kwargs': {'step':10}},
        'encoded_dim': {'args': ('encoded_dim', 5, 105), 'kwargs': {'step':10}},
        'hidden_layers': {'args': ('hidden_layers', 2, 20), 'kwargs': {'step': 2}},
        'lr': {'args': ('learning_rate', 1e-5, 1e-2), 'kwargs': {'log': True}},
        'momentum': {'args': ('momentum', 0.5, 0.9), 'kwargs': {}},
        'weight_decay': {'args': ('weight_decay', 1e-6, 1e-2), 'kwargs': {'log': True}},
        'loss_fn': {'args': ('loss_fn', ['MSE', 'FJL']), 'kwargs': {}},
        'sigma_sqr': {'args': ('sigma_sqr', 1e-6, 1e-2), 'kwargs': {'log': True}},
    }

    for k in kwargs:
        hyperparam_sample_args[k] = kwargs[k]

    # Preload data for consistency
    data = load_shuffle(data_path)

    def objective(trial):
        hyperparams = {
            'batch_sz' : trial.suggest_categorical(
                *hyperparam_sample_args['batch_sz']['args'],
                **hyperparam_sample_args['batch_sz']['kwargs'],
            ),
            'hidden_dim' : trial.suggest_int(
                *hyperparam_sample_args['hidden_dim']['args'],
                **hyperparam_sample_args['hidden_dim']['kwargs'],
            ),
            'encoded_dim' : trial.suggest_int(
                *hyperparam_sample_args['encoded_dim']['args'],
                **hyperparam_sample_args['encoded_dim']['kwargs'],
            ),
            'hidden_layers' : trial.suggest_int(
                *hyperparam_sample_args['hidden_layers']['args'],
                **hyperparam_sample_args['hidden_layers']['kwargs'],
            ),
            'lr' : trial.suggest_float(
                *hyperparam_sample_args['lr']['args'],
                **hyperparam_sample_args['lr']['kwargs'],
            ),
            'momentum' : trial.suggest_float(
                *hyperparam_sample_args['momentum']['args'],
                **hyperparam_sample_args['momentum']['kwargs'],
            ),
            'weight_decay' : trial.suggest_float(
                *hyperparam_sample_args['weight_decay']['args'],
                **hyperparam_sample_args['weight_decay']['kwargs'],
            ),
            'loss_fn' : trial.suggest_categorical(
                *hyperparam_sample_args['loss_fn']['args'],
                **hyperparam_sample_args['loss_fn']['kwargs'],
            ),
            'sigma_sqr' : trial.suggest_float(
                *hyperparam_sample_args['sigma_sqr']['args'],
                **hyperparam_sample_args['sigma_sqr']['kwargs'],
            ),
        }

        batch_sz = hyperparams['batch_sz']
        train_loader, test_dataset = normalize_split_batch(
            data,
            batch_sz,
            0.01,
            normalize=normalize,
        )

        hidden_dim = hyperparams['hidden_dim']
        encoded_dim = hyperparams['encoded_dim']
        hidden_layers = hyperparams['hidden_layers']
        model = Autoencoder(
            input_dim,
            hidden_dim,
            encoded_dim,
            mlp_blocks=hidden_layers
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=hyperparams['lr'],
            momentum=hyperparams['momentum'],
            weight_decay=hyperparams['weight_decay'],
        )
        loss = hyperparams['loss_fn']
        sigma_sqr = hyperparams['sigma_sqr']
        cov_matrix = sigma_sqr*np.identity(input_dim)
        criterion = select_loss(loss, model, sigma_sqr)
        trial_criterion = nn.HuberLoss()

        with SummaryWriter(log_dir=f'runs/optuna_trial_{trial.number}') as writer:
            stopper = EarlyStop()
            for epoch in range(max_epochs):
                epoch_loss = 0.
                model.train()
                for batch in train_loader:
                    inputs = batch[0]
                    noisy_inputs = add_gaussian_noise(inputs, np.zeros(input_dim), cov_matrix)
                    targets = inputs

                    # Forward pass
                    outputs = model(noisy_inputs)
                    loss = criterion(noisy_inputs, outputs, targets)

                    # Backpropagation and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Compute intertrial loss
                    with torch.no_grad():
                        std_loss = trial_criterion(outputs, targets)
                        epoch_loss += std_loss.item()

                # Report train loss
                epoch_loss /= len(train_loader) * sigma_sqr
                writer.add_scalar("Loss/Train", epoch_loss, epoch)
                trial.report(epoch_loss, epoch)

                # validate
                model.eval()
                with torch.no_grad():
                    inputs = test_dataset.tensors[0]
                    noisy_inputs = add_gaussian_noise(inputs, np.zeros(input_dim), cov_matrix)
                    outputs = model(noisy_inputs)
                    test_loss = trial_criterion(outputs, inputs).item()/sigma_sqr
                    writer.add_scalar("Loss/Test", test_loss, epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                # Stops training when the loss curve flattens
                if stopper.stop(epoch_loss):
                    break

            writer.add_hparams(
                hyperparams,
                {
                    'Loss/Train': epoch_loss,
                    'Loss/Test': test_loss,
                }
            )

        return test_loss

    return objective


def run_study(
    study_name,
    storage,
    input_dim,
    data_path,
    objective_kwargs,
    n_warmup_steps=10,
    n_trials=100,
    volume=1000,
    normalize=None,
):
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=NodeVolumeLimitSampler(volume),
    )

    study.optimize(
        objective_factory(data_path, input_dim, normalize=normalize, **objective_kwargs),
        n_trials=n_trials,
        n_jobs=1,
    )


def tune(
        tasks,
        study_name,
        storage,
        data_path,
        input_dim,
        volume,
        hyperparam_sample_args,
        n_trials=12,
        normalize=None,
):
    study_runner = ft.partial(
        run_study,
        study_name,
        storage,
        input_dim,
        data_path,
        hyperparam_sample_args,
        n_trials=n_trials,
        volume=volume,
        normalize=normalize,
    )

    # Run parallel studies
    procs = [Process(target=study_runner) for _ in range(0,tasks)]
    procs[0].start()

    # This is to prevent a data race between the initial processes that
    # try to setup the database
    time.sleep(2)

    for p in procs[1:]:
        p.start()
    for p in procs:
        p.join()

    study = optuna.create_study(storage=storage, study_name=study_name, load_if_exists=True)
    print(f'Best Parameters: {study.best_params}')
    print(f'Best Loss: {study.best_value}')


def tune_test():
    tasks = 2
    study_name = "DAE-Hyperparameter-Tuning-Test"
    storage = "sqlite:///dae-hp-tuning-test.db"
    data_path = '../data/silva_data_1.npy'
    input_dim = 2
    volume = 1000

    hyperparam_sample_args = {}
    tune(
        tasks,
        study_name,
        storage,
        data_path,
        input_dim,
        volume,
        hyperparam_sample_args,
        n_trials=1,
        normalize=standard_normalize,
    )


def tune1():
    tasks = 8
    study_name = "DAE-Hyperparameter-Tuning-1"
    storage = "sqlite:///dae-hp-tuning-1.db"
    data_path = '../data/silva_data_1.npy'
    input_dim = 2
    volume = 1000

    hyperparam_sample_args = {}
    tune(tasks, study_name, storage, data_path, input_dim, volume, hyperparam_sample_args)


def tune2a():
    tasks = 8
    study_name = "DAE-Hyperparameter-Tuning-2A"
    storage = "sqlite:///dae-hp-tuning-2A.db"
    data_path = '../data/gapiness_dataset_2A_Dr.npy'
    input_dim = 5
    volume = 3000

    hyperparam_sample_args = {
        'batch_sz': {'args': ('batch_sz', [16, 32, 64]), 'kwargs': {}},
        'hidden_dim': {'args': ('hidden_dim', 5, 505), 'kwargs': {'step':10}},
        'encoded_dim': {'args': ('encoded_dim', 1, 4), 'kwargs': {}},
        'hidden_layers': {'args': ('hidden_layers', 3, 27), 'kwargs': {'step': 3}},
        'lr': {'args': ('learning_rate', 1e-5, 1e-2), 'kwargs': {'log': True}},
        'momentum': {'args': ('momentum', 0.5, 0.9), 'kwargs': {}},
        'weight_decay': {'args': ('weight_decay', 1e-6, 1e-2), 'kwargs': {'log': True}},
        'loss_fn': {'args': ('loss_fn', ['MSE', 'FJL']), 'kwargs': {}},
        'sigma_sqr': {'args': ('sigma_sqr', 1e-6, 1e-2), 'kwargs': {'log': True}},
    }
    tune(tasks, study_name, storage, data_path, input_dim, volume, hyperparam_sample_args)

# TODO
# 1 Tune normalization size, maybe larger data less aggressive normalization
# 2 Look into subsampling
# 3 Maybe look at condition number
if __name__ == '__main__':
    tune_test()