from gappiness import dae
from datetime import datetime


def main():
    train_loader, test_loader = dae.load_data(
        '../data/silva_data_1.npy',
        256,
        0.1,
        normalize=dae.standard_normalize,
    )

    dae_trained = dae.train_dae(
        train_loader,
        2,
        6,
        5,
        add_noise=dae.add_gauss_noise(),
        test_loader=test_loader,
    )

    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M")
    dae_trained.save(
        f'../model/encoder_{time_stamp}',
        f'../model/decoder_{time_stamp}',
    )


main()