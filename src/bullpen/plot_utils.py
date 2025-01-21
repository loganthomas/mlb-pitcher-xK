import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# import plotly.io as pio
import scipy.stats

from bullpen.data_utils import PlayerLookup

LOOKUP = PlayerLookup()


def plot_pred_vs_target(
    X_df,
    y_df,
    preds,
    title,
    mode='static',
    savepath=None,
):
    if mode == 'static':
        plot_model = scipy.stats.linregress(preds, y_df)
        plt.scatter(preds, y_df, alpha=0.5)
        plt.plot(
            preds,
            plot_model.intercept + plot_model.slope * preds,
            'k-',
            label=f'r^2: {plot_model.rvalue**2:.3f}',
        )
        plt.xlabel('xK%')
        plt.ylabel('K%')
        plt.title(title)
        plt.legend()
        if savepath:
            plt.savefig(savepath)
        plt.show()

    if mode == 'interactive':
        data = pd.concat(
            [X_df, y_df.rename('K%'), pd.Series(preds, name='xK%')],
            axis=1,
        ).merge(LOOKUP.mapping, on=['MLBAMID', 'PlayerId'])

        fig = px.scatter(
            data,
            x='xK%',
            y='K%',
            hover_data=['Name', 'Team', 'Season'],
            trendline='ols',
            height=600,
            width=600,
            title=title,
        )
        fig.show()
        if savepath:
            # pio.write_html(fig, savepath)
            # https://kanishkegb.github.io/plotly-with-markdown/
            fig.write_html(savepath, full_html=False, include_plotlyjs='cdn')


def plot_player(player_name, X_df, y_df, preds, target_year=2024, ylim=None, savepath=None):
    ylim = [0, 0.51] if ylim is None else ylim
    data = pd.concat(
        [X_df, y_df.rename('K%'), pd.Series(preds, name='xK%')],
        axis=1,
    ).merge(LOOKUP.mapping, on=['MLBAMID', 'PlayerId'])

    player_mask = data.Name == player_name
    mlb_id, fangraphs_id = data.loc[player_mask, ['MLBAMID', 'PlayerId']].iloc[0]
    seasons = data.loc[player_mask, 'Season'].tolist()
    ks = data.loc[player_mask, 'K%'].tolist()

    target_mask = player_mask & (data.Season == target_year)
    target = (
        data.loc[target_mask, 'xK%'].item() if target_mask.sum() else 0.3
    )  # 0.3 is just a placeholder for missing data
    alpha = None if target else 0
    title = f'{player_name}\n(MLBAMID: {mlb_id} FanGraphs {fangraphs_id})'
    if not target:
        title = f'{title}\n NO TARGET DATA FOR {target_year}'

    fig, ax = plt.subplots()
    ax.plot(
        pd.to_datetime(seasons, format='%Y'),
        ks,
        marker='s',
        label='Prev Year(s) K%',
    )
    ax.scatter(
        pd.to_datetime(target_year, format='%Y'),
        target,
        marker='o',
        color='g',
        s=50,
        label=f'{target_year} xK%',
        alpha=alpha,
        zorder=99,
    )
    ax.set_ylim()
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlabel('Year')
    ax.set_ylabel('K%')
    ax.set_title(title)
    if savepath:
        plt.savefig(savepath)
    plt.show()
    print(f'xK%: {target:.4f}')
    print(f'K% : {ks}')
