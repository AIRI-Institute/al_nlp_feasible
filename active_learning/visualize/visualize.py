from typing import List, Union, Tuple
from pathlib import Path
from IPython.display import display
import warnings
from pandas import DataFrame

from .plot_figure import plot_with_confidence_interval
from .extract_metrics import collect_data

warnings.filterwarnings("ignore")


def visualize_experiments(
    paths_to_experiments: List[
        Union[str, Path]
    ],  # list with paths to experiments (str or Path)
    num_queries: int = 16,  # Number of AL queries (=config.al.num_queries + 1)
    task: str = "cls",  # "cls" / "ner" / "abs-sum"
    fill_missing: bool = False,  # whether to add an experiment with observations < `num_queries`
    prefix: str or None = "test",  # "test" or "eval" or None
    quantiles: Union[list, tuple] = None,  # quantiles to use if loading Tracin results
    log_errors: bool = False,  # whether to log the errors for unplotted experiments
    metric_name: str = None,  # if there is a specific metric you want to visualize (e.g. with abs-sum)
    metric_file_names: Union[
        List[str], Tuple[str]
    ] = None,  # if wanna use distinct files with metrics
    display_df_head: bool = False,  # whether we want to display the head of the `df` with experiments
    query_condition: str = None,  # query condition for observations from `df` which we want to plot
    columns_to_group_by=(
        "al_type",
        "strategy",
        "acquisition",
        "successor",
        "target",
    ),  # columns to groupby to separate different experiments
    save_name=None,  # path to save the plotted figure
    sleeve_type="conf",  # type of sleeve, either `conf` (confidence interval) or any other for ±std
    title="",  # plot title
    columns=None,  # how to name columns (otherwise names automatically)
    index=None,  # must have length equal to num_queries
    colors=None,  # colors for sleeves
    line_dash_sequence=None,  # how to plot different lines
    legend_font_size=20,  # size of the legend
    labels_font_size=15,  # size of the xlabel / ylabel text
    **kwargs  # other kwargs for px.line
):
    df = collect_data(
        paths_to_experiments,
        num_queries=num_queries,
        task=task,
        fill_missing=fill_missing,
        prefix=prefix,
        quantiles=quantiles,
        log_errors=log_errors,
        metric_name=metric_name,
        metric_file_names=metric_file_names,
    )
    if display_df_head:
        display(df.head())

    if query_condition is not None:
        df = df.query(query_condition)
    fig = plot_with_confidence_interval(
        df,
        columns_to_group_by=list(columns_to_group_by),
        save_name=save_name,
        metric_name=metric_name,
        task=task,
        sleeve_type=sleeve_type,
        title=title,  # plot title
        columns=columns,  # how to name columns (otherwise names automatically)
        index=index,  # must have length equal to num_queries
        colors=colors,  # colors for sleeves
        line_dash_sequence=line_dash_sequence,  # how to plot different lines
        legend_font_size=legend_font_size,  # size of the legend
        labels_font_size=labels_font_size,  # size of the xlabel / ylabel text
        **kwargs  # other kwargs for px.line
    )
    return fig


def plot_query(
    df: DataFrame,
    query_condition: str = None,  # query condition for observations from `df` which we want to plot
    **plot_with_confidence_interval_kwargs  # kwargs for `plot_with_confidence_interval`
):
    if query_condition is not None:
        df = df.query(query_condition)
    else:
        df = df.copy(deep=True)
    fig = plot_with_confidence_interval(df, **plot_with_confidence_interval_kwargs)
    return fig
