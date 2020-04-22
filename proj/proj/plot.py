import pandas as pd
import matplotlib.pyplot as plt


def df_group_violins(ax: plt.Axes,
                     df: pd.DataFrame,
                     variable_names: list,
                     group_by: dict,
                     violin_args: dict = {},
                     group_legend_names: dict = {},):
    groups = df.groupby(**group_by)
    fig: plt.Figure
    ax: plt.Axes

    violins = {}
    positions = list(range(1, len(variable_names) + 1))
    for group_name, group in groups:
        if group_name in group_legend_names:
            group_name = group_legend_names[group_name]
        else:
            group_legend_names[group_name] = group_name

        values = []
        for var_name in variable_names:
            values.append(group[var_name].values.reshape(-1))

        violins[group_name] = ax.violinplot(values, positions, **violin_args)

    if violin_args.get('vert', True):
        ax.set_xticks(positions)
        ax.set_xticklabels(variable_names)
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels(variable_names)

    legend_handles = [v['bodies'][0] for v in violins.values()]
    legend_labels = group_legend_names.values()
    ax.legend(legend_handles, legend_labels)
