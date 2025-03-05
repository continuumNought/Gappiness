import optuna
import torch
import time
import functools as ft
from torch.utils.tensorboard import SummaryWriter
from gappiness.loss import FastJacobianRegularizedLoss, MSELossWrapper
from gappiness.model import Autoencoder
from gappiness.data import load_data, standard_normalize
from gappiness.noise import add_2d_gaussian_noise
from multiprocessing import Process
from optuna.samplers import RandomSampler


class ConstrainedSampler(RandomSampler):
    def __init__(self, base_sampler=None):
        super().__init__()
        self.base_sampler = base_sampler or RandomSampler()

    def sample_independent(self, study, trial, param_name, param_distribution):
        while True:
            value = self.base_sampler.sample_independent(study, trial, param_name, param_distribution)

            hidden_dim = trial.params.get("hidden_dim", value if param_name == "hidden_dim" else None)
            encoded_dim = trial.params.get("encoded_dim", value if param_name == "encoded_dim" else None)
            hidden_layers = trial.params.get("hidden_layers", value if param_name == "hidden_layers" else None)

            if hidden_dim is not None and encoded_dim is not None and hidden_dim < encoded_dim:
                continue

            if hidden_dim is not None and hidden_layers is not None and hidden_dim*hidden_layers > 1000:
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


def objective_factory(data_path, input_dim, max_epochs=100):
    def objective(trial):
        hyperparams = {
            'batch_sz' : trial.suggest_categorical('batch_sz', [16,32,64]),
            'hidden_dim' : trial.suggest_int('hidden_dim', 5, 205, step=10),
            'encoded_dim' : trial.suggest_int('encoded_dim', 5, 105, step=10),
            'hidden_layers' : trial.suggest_int('hidden_layers', 2, 20, step=2),
            'lr' : trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'momentum' : trial.suggest_float('momentum', 0.5, 0.9),
            'weight_decay' : trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'loss_fn' : trial.suggest_categorical('loss_fn', ['MSE', 'FJL']),
            'sigma_sqr' : trial.suggest_float('sigma_sqr', 1e-6, 1e-2, log=True),
        }

        batch_sz = hyperparams['batch_sz']
        train_loader, test_dataset = load_data(
            data_path,
            batch_sz,
            0.01,
            normalize=standard_normalize,
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
        cov_matrix = ((sigma_sqr, 0.0), (0.0, sigma_sqr))
        criterion = select_loss(loss, model, sigma_sqr)

        with SummaryWriter(log_dir=f'runs/optuna_trial_{trial.number}') as writer:
            stopper = EarlyStop()
            for epoch in range(max_epochs):
                epoch_loss = 0.
                model.train()
                for batch in train_loader:
                    inputs = batch[0]
                    noisy_inputs = add_2d_gaussian_noise(inputs, cov_matrix=cov_matrix)
                    targets = inputs

                    # Forward pass
                    outputs = model(noisy_inputs)
                    loss = criterion(noisy_inputs, outputs, targets)
                    epoch_loss += loss.item()

                    # Backpropagation and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Report train loss
                epoch_loss /= len(train_loader)
                writer.add_scalar("Loss/Train", epoch_loss, epoch)
                trial.report(epoch_loss, epoch)

                # validate
                model.eval()
                with torch.no_grad():
                    inputs = test_dataset.tensors[0]
                    noisy_inputs = add_2d_gaussian_noise(inputs, cov_matrix=cov_matrix)
                    outputs = model(noisy_inputs)
                    test_loss = criterion(noisy_inputs, outputs, inputs).item()
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


def run_study(study_name, storage, input_dim, data_path, n_warmup_steps=10, n_trials=100):
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=ConstrainedSampler(),
    )

    study.optimize(
        objective_factory(data_path, input_dim),
        n_trials=n_trials,
        n_jobs=1,
    )


def main():
    tasks = 2
    study_name = "DAE-Hyperparameter-Tuning"
    storage = "sqlite:///dae-hp-tuning.db"
    data_path = '../data/silva_data_1.npy'
    input_dim = 2

    study_runner = ft.partial(
        run_study,
        study_name,
        storage,
        input_dim,
        data_path,
        n_trials=2,
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


if __name__ == '__main__':
    main()