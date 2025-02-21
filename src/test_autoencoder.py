import functools as ft
from gappiness import dae
from datetime import datetime


def main():
    train_loader, test_dataset = dae.load_data(
        '../data/silva_data_1.npy',
        32,
        0.01,
        normalize=dae.standard_normalize,
    )

    dae_trained = dae.train_dae(
        train_loader,
        2,
        10,
        6,
        epochs=1,
        add_noise=ft.partial(dae.add_2d_gaussian_noise, cov_matrix=((0.01,0.),(0.,0.01))),
        test_dataset=test_dataset,
    )

    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M")
    dae_trained.save(
        f'../model/encoder_{time_stamp}',
        f'../model/decoder_{time_stamp}',
    )


main()