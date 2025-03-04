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


def select_loss(loss, model, sigma_sqr):
    if loss == 'FJL':
        return FastJacobianRegularizedLoss(model, sigma_sqr=sigma_sqr)

    return MSELossWrapper()


def objective_factory(data_path, input_dim, max_epochs=100):
    def objective(trial):
        hyperparams = {
            'batch_sz' : trial.suggest_categorical('batch_sz', [16,32,64]),
            'hidden_dim' : trial.suggest_int('hidden_dim', 5, 200, step=5),
            'encoded_dim' : trial.suggest_int('encoded_dim', 5, 100, step=5),
            'hidden_layers' : trial.suggest_int('hidden_layers', 1, 20),
            'lr' : trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'momentum' : trial.suggest_float('momentum', 0.5, 0.99),
            'weight_decay' : trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'loss_fn' : trial.suggest_categorical('loss_fn', ['MSE', 'FJL']),
            'sigma_sqr' : trial.suggest_float('sigma_sqr', 1e-6, 1.0, log=True),
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
                    test_loss = criterion(noisy_inputs, outputs, inputs).item()/len(inputs)
                    writer.add_scalar("Loss/Test", test_loss, epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

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

    study = optuna.create_study(storage=storage, study_name=study_name)
    print(f'Best Parameters: {study.best_params}')
    print(f'Best Loss: {study.best_value}')


if __name__ == '__main__':
    main()