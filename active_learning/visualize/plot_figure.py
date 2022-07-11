import numpy as np
import plotly.express as px
from scipy.stats import t
from typing import Union


PRETRAINED_NAMES_MAPPING = {
    "distilbert": "DistilBERT",
    "bert-base": "BERT base",
    "bert-large": "BERT large",
    "google/electra-base-discriminator": "ELECTRA base",
    "roberta-base": "RoBERTa base",
    "roberta-large": "RoBERTa large",
    "xlnet-base": "XLNet base",
    "xlnet-large": "XLNet large",
    "distilrobert": "DistilRoBERTa",
    "microsoft/": "DeBERTa",
    "facebook/bart-base": "BART base",
    "facebook/bart-large": "BART large",
}

AL_EXPERIMENT_TYPES_MAPPING = {
    "one": {"cls": "LC", "ner": "MNLP", "abs-sum": "Seq. score"},
    "asm": "A-S M",
    "astm": "PLASM",
    "plasm": "PLASM",
    "full": "Full data",
}

COLORS = [
    "rgb(50, 150, 200)",
    "rgb(30, 180, 30)",
    "rgb(128, 0, 128)",
    "rgb(255, 165, 0)",
    "rgb(200, 128, 150)",
    "rgb(128, 128, 128)",
    "rgb(20, 200, 250)",
    "rgb(255, 15, 255)",
    "rgb(60, 180, 120)",
    "rgb(128, 0, 0)",
]


def _make_sleeve_color(color):
    return "rgba" + color[3:-1] + ", 0.2)"


def _sub_name(name):
    for pretrained_name, mapping in PRETRAINED_NAMES_MAPPING.items():
        if name.startswith(pretrained_name):
            return mapping
    return name


def _sub_al_experiment_type(name, task="cls"):
    for al_exp_type, mapping in AL_EXPERIMENT_TYPES_MAPPING.items():
        if name == al_exp_type:
            if isinstance(mapping, dict):
                return mapping[task]
            return mapping
    return name.title()


def annotate_line(columns_to_group_by, columns_to_substitute):
    """
    Function to generate automatic annotation for each line.
    `transformations_list` will consist of triples
    <initial_text, function_to_transform_value_in_this_column, idx_of_this_column>
    """
    transformations_list = []
    if "id_experiment" in columns_to_group_by:
        idx = columns_to_group_by.index("id_experiment")
        transformations_list.append(["exp. ", None, idx])
    if "metric_name" in columns_to_group_by:
        idx = columns_to_group_by.index("metric_name")
        transformations_list.append(["metric: ", None, idx])
    if "al_type" in columns_to_group_by:
        idx = columns_to_group_by.index("al_type")
        transformations_list.append(["", _sub_al_experiment_type, idx])
    if "strategy" in columns_to_group_by:
        idx = columns_to_group_by.index("strategy")
        transformations_list.append(["strat. ", None, idx])
    if "acquisition" in columns_to_group_by:
        idx = columns_to_group_by.index("acquisition")
        transformations_list.append(["acq. ", _sub_name, idx])
    if "target" in columns_to_group_by:
        idx = columns_to_group_by.index("target")
        transformations_list.append(["succ. ", _sub_name, idx])
    if "ups" in columns_to_group_by:
        idx = columns_to_group_by.index("ups")
        transformations_list.append(["", None, idx])
    if "framework" in columns_to_group_by:
        idx = columns_to_group_by.index("framework")
        transformations_list.append(["", None, idx])
    if "deleted" in columns_to_group_by:
        idx = columns_to_group_by.index("deleted")
        transformations_list.append(["del. ", str, idx])

    new_columns = []
    for multicolumn in columns_to_substitute:
        result = ""
        for (pre_text, func, idx) in transformations_list:
            text = func(multicolumn[idx]) if func is not None else multicolumn[idx]
            result += pre_text + str(text) + ", "
        result = result[:-2]  # Remove the last comma and space
        new_columns.append(result)
    return new_columns


def plot_with_confidence_interval(
    df,
    columns_to_group_by=(
        "al_type",
        "strategy",
        "acquisition",
        "successor",
        "target",
    ),  # columns to groupby
    save_name="",  # how to save the img. if empty, no save is done
    metric_name=None,  # name for the metric on the y-scale
    task="cls",  # "cls" / "ner" / "abs-sum"
    sleeve_type="conf",  # how to plot the variance sleeve. either "conf" for conf interval or any other for ±std
    count=5,  # how many seeds we have. needs to be removed since we can have different num for different models
    title="",  # plot title
    columns=None,  # how to name columns (otherwise names automatically)
    index=None,  # must have length equal to num_queries
    colors=None,  # colors for lines and sleeves
    line_dash_sequence=None,  # how to plot different lines
    legend_font_size=20,  # size of the legend
    labels_font_size=15,  # size of the x_label / y_label text
    x_label: str = None,  # name of x axis
    y_label: str = None,  # name of y axis
    remove_whitespace: bool = True,  # whether to remove whitespaces around the figure when saving it
    x_title_standoff: Union[
        int, float
    ] = None,  # if not None, the margin between x-label and the axis
    y_title_standoff: Union[
        int, float
    ] = None,  # if not None, the margin between y-label and the axis
    **kwargs,  # other kwargs for px.line
):
    """
    sleeve_type == "conf" or "std". if "conf", `count` should be specified
    """
    if metric_name is None:
        if task == "cls":
            metric_name = "Performance, Accuracy"
            df.rename(columns={"Performance, F1": metric_name}, inplace=True)
        elif task == "ner":
            metric_name = "Performance, F1"
        elif task == "abs-sum":
            metric_name = "Performance, Rouge-L"
        else:
            raise NotImplementedError
    else:
        metric_name = f"Performance, {metric_name.title()}"
    # Remove potentially redundant columns
    for column in ["id_experiment", "deleted", "seed"]:
        if (column in df) and (column not in columns_to_group_by):
            df.drop(columns=column, inplace=True)
    # Group by the specified columns
    groupby = df.groupby(list(columns_to_group_by))
    df_mean = groupby.mean().T
    df_std = groupby.std().T

    if sleeve_type == "conf":
        df_count = groupby.count().T
        columns_to_save = [x for x in df_count.index if x.startswith("f1")]
        df_count = df_count.loc[columns_to_save]
        # Calculate components for conf interval
        conf_lev_vals = t.ppf(0.975, df_count)
        std_by_conf_val = df_std * conf_lev_vals
        counts_sqrt = df_count ** (1 / 2)
        # Calculate lower and upper bounds
        df_lb = df_mean - std_by_conf_val / counts_sqrt
        df_ub = df_mean + std_by_conf_val / counts_sqrt
    else:
        df_lb = df_mean - df_std
        df_ub = df_mean + df_std

    # Name columns if they are not specified
    if columns is None:
        columns = annotate_line(columns_to_group_by, df_mean.columns)
    df_mean.columns = df_ub.columns = df_lb.columns = columns

    # Get percent of labeled data so far if not specified
    if index is None:
        index = np.arange(1, len(df_mean) + 1)
        if task == "ner":
            index *= 2
        elif task == "abs-sum":
            index *= 10
    df_mean.index = df_lb.index = df_ub.index = index
    # Label axes
    if x_label is None:
        if task == "abs-sum":
            x_label = "Num. labeled instances"
        else:
            x_label = "Labeled Data, %"
    if y_label is None:
        y_label = metric_name

    if line_dash_sequence is None:
        line_dash_sequence = ["dash", "dashdot", "longdashdot", "solid", "longdash"]

    # Reset index, melt and rename the column
    df_to_plot = df_mean.reset_index()
    df_to_plot = df_to_plot.melt(id_vars=["index"], value_name=metric_name)
    df_to_plot = df_to_plot.rename(columns={"index": x_label, metric_name: y_label})

    # Add sleeve colors if neither they nor colors are specified
    if colors is None:
        colors = COLORS
    sleeve_colors = list(map(_make_sleeve_color, colors))

    # Plot the lines
    fig = px.line(
        df_to_plot,
        x=x_label,
        y=y_label,
        line_dash="variable",
        line_dash_sequence=line_dash_sequence,
        title=title,
        color="variable",
        color_discrete_sequence=colors,
        **kwargs,
    )

    # Plot sleeves
    for i, name in enumerate(df_mean.columns):
        # Upper bound
        fig.add_scatter(
            x=df_ub.index,
            y=df_ub.iloc[:, i],
            name=name + "_upper_bound",
            marker=dict(color="#444", size=0),
            line=dict(width=0),
            showlegend=False,
            mode="lines",
        )
        # Lower bound
        fig.add_scatter(
            x=df_lb.index,
            y=df_lb.iloc[:, i],
            name=name + "_lower_bound",
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor=sleeve_colors[i],
            fill="tonexty",
            showlegend=False,
            mode="lines",
        )

    # Update legend position & font
    labels_font_size = 15 if labels_font_size is None else labels_font_size
    fig.update_layout(
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            font=dict(size=legend_font_size),
        ),
        font=dict(size=labels_font_size),
        legend_title_text=None,
    )
    # Title standoff
    if x_title_standoff is not None:
        fig.update_xaxes(title_standoff=x_title_standoff)
    if y_title_standoff is not None:
        fig.update_yaxes(title_standoff=y_title_standoff)

    # Save
    if save_name != "" and save_name is not None:
        if remove_whitespace:
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
            )
        fig.write_image(save_name)

    return fig
