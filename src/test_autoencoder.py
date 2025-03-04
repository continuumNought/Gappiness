import matplotlib.pyplot as plt
import numpy as np
import functools as ft
from gappiness import dae
from datetime import datetime
from scipy.spatial.distance import pdist
from scipy.stats import chi2


#np.random.seed(0x2034578)


def get_cov_matrix(path_):
    data = np.load(path_)
    data_n = dae.standard_normalize(data)
    return np.cov(data_n, rowvar=False)


def get_scaled_cov(path_):
    # TODO for wider dataset use svd to find principal directions of covariance matrix
    data = np.load(path_)
    data_n = dae.standard_normalize(data)
    data_cov =  np.cov(data_n, rowvar=False)

    subset = data_n[np.random.choice(data_n.shape[0], size=1000, replace=False)]
    median_dist = np.median(pdist(subset))
    df = 2
    chi2_median = np.sqrt(chi2.ppf(0.5, df))
    scale = median_dist / chi2_median

    return data_cov * (scale**2)


def analyze_data(path_):
    data = np.load(path_)
    data_n = dae.standard_normalize(data)
    data_n_cov = np.cov(data_n, rowvar=False)

    print(f'shape: {data_n.shape}')
    print(f'covariance: {data_n_cov}')

    subset = data_n[np.random.choice(data_n.shape[0], size=1000, replace=False)]
    num_bins = 70
    bin_edges = np.linspace(0, 7, num_bins + 1)
    bin_counts = np.zeros(num_bins)

    for i in range(0,subset.shape[0]-1):
        hist_counts, _ = np.histogram(
            np.sqrt(np.sum((subset[i] - subset[i + 1:]) ** 2, axis=1)),
            bins=bin_edges
        )
        bin_counts += hist_counts  # Accumulate counts

    bin_relative_freq = bin_counts / bin_counts.sum()

    # Plot the histogram manually using bar chart
    plt.bar(
        bin_edges[:-1],
        bin_relative_freq,
        width=bin_edges[1] - bin_edges[0],
        edgecolor="black",
        alpha=0.7
    )

    # Labels and title
    plt.xlabel("Distance")
    plt.ylabel("Relative Frequency")
    plt.title("Histogram of Pairwise Distances")

    # Show plot
    plt.show()


def main():
    path_ = '../data/silva_data_1.npy'
    # analyze_data(path_)
    train_loader, test_dataset = dae.load_data(
        path_,
        64,
        0.01,
        normalize=dae.standard_normalize,
    )

    # data_n_cov = get_scaled_cov(path_)
    data_n_cov = get_cov_matrix(path_)
    sigma_sqr = 0.001

    dae_trained = dae.train_dae(
        train_loader,
        2,
        100,
        20,
        epochs=10,
        add_noise=ft.partial(
            dae.add_2d_gaussian_noise,
            cov_matrix=sigma_sqr*data_n_cov,
        ),
        test_dataset=test_dataset,
        hidden_layers=5,
        criterion_factory=lambda m: dae.FastJacobianRegularizedLoss(m, sigma_sqr=sigma_sqr)
    )

    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M")
    dae_trained.save(
        f'../model/encoder_{time_stamp}',
        f'../model/decoder_{time_stamp}',
    )


main()