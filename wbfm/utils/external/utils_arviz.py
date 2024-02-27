import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_ts(idata, y='y', y_hat='y', num_samples=100):
    """
    My implementation of the plot_ts function from ArviZ, which doesn't work for me.

    Uses plotly

    Parameters
    ----------
    idata

    Returns
    -------

    """

    # Build everything in a big dataframe, which will be used to plot everything
    y_obs = idata.observed_data.y

    # Get a sample from the large matrix of predictions
    # Original shape: (num_chains, num_draws, num_timesteps)
    # Target shape: (num_samples, num_timesteps)
    # First flatten the first two dimensions
    y_hat = idata.posterior_predictive.y
    y_hat = np.array(y_hat).reshape(-1, y_hat.shape[2])
    # Take a random sample from both num_chains and num_draws
    idx = np.random.choice(y_hat.shape[0], num_samples)
    y_hat = y_hat[idx]

    # Combine everything into a single dataframe
    df = pd.DataFrame(y_hat.T)
    df['observed'] = y_obs

    # Plot
    fig = go.Figure()
    # Average of the samples
    fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, :-1].mean(axis=1), mode='lines', name='y_hat',
                             line=dict(color='gray', width=2)))
    # 95% CI
    fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, :-1].quantile(0.025, axis=1), mode='lines', name='y_hat',
                             line=dict(color='gray', width=1), showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, :-1].quantile(0.975, axis=1), mode='lines', name='y_hat',
                             line=dict(color='gray', width=1), fill='tonexty',
                             showlegend=False))

    # Observed
    fig.add_trace(go.Scatter(x=df.index, y=df['observed'], mode='lines', name='observed',
                             line=dict(color='black', width=2)))

    fig.show()
    return fig
