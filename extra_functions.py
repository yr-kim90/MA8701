import numpy as np
import matplotlib.pyplot as plt
from group_lasso import GroupLasso
from sklearn.model_selection import cross_val_score


def standardize(X, y):
    # Standardize X to have mean = 0 std = 1
    # Standardize y to have mean = 0
    X_scaled = (X-np.mean(X, axis=0))/np.std(X, axis=0)
    y_scaled = y-np.mean(y)
    return X_scaled, y_scaled


def plot_data(df):
    """Convenience function for plotting dataframes"""
    fig, axs = plt.subplots(len(df.columns), figsize=(6, 4*len(df.columns)))
    axs = axs.flatten()
    for ax, c in zip(axs, df.columns):
        ax.plot(df[c])
        ax.set_ylabel(c)
    return fig


def plot_coefficients(beta, alpha, alpha_opt=10, name=None):
    # Plotting regression coefficients vs lambda
    fig, ax = plt.subplots(1)
    ax.plot(np.log10(alpha), beta.T, '-')
    ax.plot(np.log10(alpha_opt)*np.array([1, 1]),
            [np.min(beta), np.max(beta)], 'k--')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\beta$')
    ax.set_title(name)
    return fig


def plot_CV_MSE(alpha_vals, mse, alpha_opt, name=None):
    mse_mean = np.mean(mse, axis=1)
    mse_std = np.std(mse, axis=1)
    fig, ax = plt.subplots(1)
    ax.errorbar(np.log10(alpha_vals), mse_mean, mse_std)
    ax.plot(np.log10([alpha_opt, alpha_opt]), [0, np.max(mse)], 'k--')
    ax.set_xlabel(r'log($\lambda$)')
    ax.set_ylabel('MSE')
    ax.set_title(name)
    print(f"Optimal value of lambda is: {np.round(alpha_opt, 3)}")
    return fig


def feature_selection_cv(X, y, alpha_vals, reg, group_reg=None):
    n_alpha = len(alpha_vals)
    beta = np.zeros((X.shape[1], n_alpha))
    cv_outs = []
    min_cv = np.inf
    for i, alpha in enumerate(alpha_vals):
        if type(reg) == GroupLasso:   # Handling group lasso
            reg.group_reg = group_reg*alpha
            reg.l1_reg = (1-group_reg)*alpha
        else:
            reg.alpha = alpha

        reg.fit(X, y)
        beta[:, i] = reg.coef_.reshape(-1)
        cv_out = cross_val_score(reg, X, y, cv=10,
                                 scoring='neg_mean_squared_error')
        cv_outs.append(cv_out)
        if -cv_out.mean() < min_cv:
            min_cv = -cv_out.mean()
            min_alpha = alpha

    beta_best = beta[:, alpha_vals == min_alpha]
    return beta, beta_best, cv_outs, min_alpha


def bootstrap_loop(X, y, alpha_vals, reg, b=20, N_samples=100, random_state=0,
                   *args, **kwargs):
    betas = np.zeros((N_samples, X.shape[1]))
    rng = np.random.default_rng(random_state)
    for n in range(N_samples):
        index_vector = np.arange(len(y))
        boot_index = rng.choice(index_vector, size=b)
        X_sample = X[boot_index, :]
        y_sample = y[boot_index]
        _, beta_best, _, _ = feature_selection_cv(X_sample, y_sample,
                                                  alpha_vals, reg,
                                                  *args, **kwargs)
        betas[n, :] = beta_best.reshape(-1)
    return betas
