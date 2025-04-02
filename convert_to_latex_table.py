from clustpy.utils.evaluation import evaluation_df_to_latex_table
import os
import pandas as pd
import numpy as np


def evaluation_df_to_latex_table_percentages(df: pd.DataFrame, output_path: str, use_std: bool = True, best_in_bold: bool = True,
                                 second_best_underlined: bool = True, color_by_value: str = None,
                                 higher_is_better: list = None, in_percent: int = True,
                                 decimal_places: int = 1, percentages: list = None) -> None:
    """
    Convert the resulting dataframe of an evaluation into a latex table.
    Note that the latex package booktabs is required, so usepackage{booktabs} must be included in the latex file.
    This method will only consider the mean values. Therefore, note that "mean" must be included in the aggregations!
    If "std" is also contained in the dataframe (and use_std is True) this value will also be added by using plusminus.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas dataframe. Can also be a string that contains the path to the saved dataframe
    output_path : std
        The path were the resulting latex table text file will be stored
    use_std : bool
        Defines if the standard deviation (std) should also be added to the latex table (default: True)
    best_in_bold : bool
        Print best value for each combination of dataset and metric in bold.
        Note, that the latex package bm is used, so usepackage{bm} must be included in the latex file (default: True)
    second_best_underlined : bool
        Print second-best value for each combination of dataset and metric underlined (default: True)
    color_by_value : str
        Define the color that should be used to indicate the difference between the values of the metrics.
        Uses colorcell, so usepackage{colortbl} or usepackage[table]{xcolor} must be included in the latex file.
        Can be 'blue' for example (default: None)
    higher_is_better : list
        List with booleans. Each value indicates if a high value for a certain metric is better than a low value.
        The length of the list must be equal to the number of different metrics.
        If None, it is always assumed that a higher value is better, except for the runtime (default: None)
    in_percent : bool
        If true, all values, except n_clusters and runtime, will be converted to percentages -> all values will be multiplied by 100 (default: True)
    decimal_places : int
        Number of decimal places that should be used in the latex table (default: 1)
    """
    # Load dataframe
    assert percentages is not None, "percentages can not be None. or use evaluation_df_to_latex_table function"
    assert type(df) == pd.DataFrame or type(df) == str, "Type of df must be pandas DataFrame or string (path to file)"
    if type(df) == str:
        df_file = open(df, "r").readlines()
        multiple_datasets = df_file[2].split(",")[0] != "0"
        df = pd.read_csv(df, index_col=[0, 1] if multiple_datasets else [0], header=[0, 1])
    else:
        multiple_datasets = isinstance(df.index, pd.MultiIndex)
    # Get main information from dataframe
    if multiple_datasets:
        datasets = list(dict.fromkeys([s[0] for s in df.index]))
        std_contained = "std" in [s[1] for s in df.index]
    else:
        datasets = [None]
        std_contained = "std" in [s for s in df.index]
    algorithms = list(dict.fromkeys([s[0] for s in df.keys()]))
    metrics = list(dict.fromkeys([s[1] for s in df.keys()]))
    assert higher_is_better is None or len(higher_is_better) == len(
        metrics), "Length of higher_is_better and the number of metrics does not match. higher_is_better = {0} (length {1}), metrics = {2} (length {3})".format(
        higher_is_better, len(higher_is_better), metrics, len(metrics))
    # Write output
    with open(output_path, "w") as f:
        # Write standard table
        f.write(
            "\\begin{table}\n\\centering\n\\caption{TODO}\n\\resizebox{1\\textwidth}{!}{\n\\begin{tabular}{l|")
        if multiple_datasets:
            f.write("l|" + "c" * len(algorithms) + "}\n\\toprule\n\\textbf{Dataset} & ")
        else:
            f.write("c" * len(algorithms) + "}\n\\toprule\n")
        f.write("\\textbf{Metric} & " + " & ".join(algorithms) + "\\\\\n\\midrule\n")
        # Write values into table
        for j, d in enumerate(datasets):
            # Check if underscore in dataset name
            if d is not None:
                if os.name != "nt":
                    d = d.replace("_", "\\_")
            for i, m in enumerate(metrics):
                # Check if underscore in metric name
                if os.name != "nt":
                    m = m.replace("_", "\\_")
                # Check if a higher value is better for this metric
                metric_is_higher_better = (m != "runtime") if higher_is_better is None else higher_is_better[i]
                # Write name of dataset and metric
                if multiple_datasets:
                    if i == 0:
                        to_write = d + " & " + m
                    else:
                        to_write = "& " + m
                else:
                    to_write = m
                # Get all values from the experiments (are stored separately to calculated min values)
                all_values = []
                for a in algorithms:
                    if multiple_datasets:
                        mean_value = df[a, m][d, "mean"]
                    else:
                        mean_value = df[a, m]["mean"]
                    if in_percent and m not in ["n_clusters", "runtime"]:
                        mean_value *= 100
                    mean_value = round(mean_value, decimal_places)
                    all_values.append(mean_value)
                all_values_sorted = np.unique(all_values)  # automatically sorted
                for k, a in enumerate(algorithms):
                    # Check if underscore in algorithm name
                    if os.name != "nt":
                        a = a.replace("_", "\\_")
                    mean_value = all_values[k]
                    # If standard deviation is contained in the dataframe, information will be added
                    if use_std and std_contained:
                        if multiple_datasets:
                            std_value = df[a, m][d, "std"]
                        else:
                            std_value = df[a, m]["std"]
                        if in_percent and m not in ["n_clusters", "runtime"]:
                            std_value *= 100
                        std_value = round(std_value, decimal_places)
                        value_write = "$" + str(mean_value) + " \\pm " + str(std_value) + "$"
                    else:
                        value_write = "$" + str(mean_value) + "$"
                    # Optional: Write best value in bold and second best underlined
                    if best_in_bold and ((mean_value == all_values_sorted[-1] and metric_is_higher_better) or (
                            mean_value == all_values_sorted[0] and not metric_is_higher_better)):
                        value_write = "\\bm{" + value_write + "}"
                    elif second_best_underlined and (
                            (mean_value == all_values_sorted[-2] and metric_is_higher_better) or (
                            mean_value == all_values_sorted[1] and not metric_is_higher_better)):
                        value_write = "\\underline{" + value_write + "}"
                    # Optional: Color cells by value difference
                    if color_by_value is not None:
                        if all_values_sorted[-1] != all_values_sorted[0]:
                            color_saturation = round((mean_value - all_values_sorted[0]) / (
                                    all_values_sorted[-1] - all_values_sorted[0]) * 65) + 5  # value between 5 and 70
                        else:
                            color_saturation = 0
                        assert type(color_saturation) is int, "color_saturation must be an int but is {0}".format(
                            type(color_saturation))
                        value_write = "\\cellcolor{" + color_by_value + "!" + str(color_saturation) + "}" + value_write
                    to_write += " & " + value_write
                to_write += "\\\\\n"
                f.write(to_write)
            if j != len(datasets) - 1:
                f.write("\\midrule\n")
            else:
                f.write("\\bottomrule\n\\end{tabular}}\n\\end{table}")


def main():
    """
    df = "C:/Users/Chris/Desktop/Uni/Masterarbeit/Experiment_results/MyBenchmark_Semisupervised_loss/semisupervised_init_small_ff_512_256_128_10/Results/result.csv"
    df = "C:/Users/Chris/Desktop/Uni/Masterarbeit/Experiment_results/MyBenchmark_Semisupervised_init/semisupervised_init_small_ff_512_256_128_10/Results/result.csv"
    df = "C:/Users/Chris/Desktop/Uni/Masterarbeit/Experiment_results/MyBenchmark_Semisupervised_init_simple/semisupervised_init_small_ff_512_256_128_10/Results/result.csv"
    df = "C:/Users/Chris/Desktop/Uni/Masterarbeit/Experiment_results/MyBenchmark_Semisupervised_init_simple_no_recluster/semisupervised_init_small_ff_512_256_128_10/Results/result.csv"
    df = "C:/Users/Chris/Desktop/Uni/Masterarbeit/Experiment_results/MyBenchmark_Semisupervised_init_simple_enrc_recluster/semisupervised_init_small_ff_512_256_128_10/Results/result.csv"
    df = "C:/Users/Chris/Desktop/Uni/Masterarbeit/Code/ClustPy_semi_supervised/MyBenchmark_Semi_loss_delayed/semi_small_ff_512_256_128_10/Results/result.csv"

    """
    df = "C:/Users/Chris/Desktop/Uni/Masterarbeit/Experiment_results/MyBenchmark_Semisupervised_init_enrc_recluster_var_labels/10_percent_labeled/Results/result.csv"
    output_path = df[:-3] + "tex"
    evaluation_df_to_latex_table(df, output_path)


if __name__ == "__main__":
    main()
