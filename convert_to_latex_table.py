from clustpy.utils.evaluation import evaluation_df_to_latex_table


def main():
    df = "C:/Users/Chris/Desktop/Uni/Masterarbeit/Experiment_results/MyBenchmark_Semisupervised_loss/semisupervised_init_small_ff_512_256_128_10/Results/result.csv"
    df = "C:/Users/Chris/Desktop/Uni/Masterarbeit/Experiment_results/MyBenchmark_Semisupervised_init/semisupervised_init_small_ff_512_256_128_10/Results/result.csv"
    df = "C:/Users/Chris/Desktop/Uni/Masterarbeit/Experiment_results/MyBenchmark_Semisupervised_init_simple/semisupervised_init_small_ff_512_256_128_10/Results/result.csv"
    df = "C:/Users/Chris/Desktop/Uni/Masterarbeit/Experiment_results/MyBenchmark_Semisupervised_init_simple_no_recluster/semisupervised_init_small_ff_512_256_128_10/Results/result.csv"
    df = "C:/Users/Chris/Desktop/Uni/Masterarbeit/Experiment_results/MyBenchmark_Semisupervised_init_simple_enrc_recluster/semisupervised_init_small_ff_512_256_128_10/Results/result.csv"
    df = "C:/Users/Chris/Desktop/Uni/Masterarbeit/Code/ClustPy_semi_supervised/MyBenchmark_Semi_loss_delayed/semi_small_ff_512_256_128_10/Results/result.csv"
    """
    list_of_percentages = [0,10,30,50,70,90,100]
    for percentage in list_of_percentages:
        df = f"C:/Users/Chris/Desktop/Uni/Masterarbeit/Experiment_results/MyBenchmark_Semisupervised_init_enrc_recluster_var_labels/{percentage}_percent_labeled/Results/result.csv"
        df = f"C:/Users/Chris/Desktop/Uni/Masterarbeit/Code/ClustPy_semi_supervised/MyBenchmark_Semi_loss_diff_label/semi_small_ff_labels_512_256_128_10/Results/result_MNIST_semi_small_ff_labels_512_256_128_10_{percentage}.csv"
        output_path = df[:-3]+"tex"
        evaluation_df_to_latex_table(df, output_path)
    """
    output_path = df[:-3] + "tex"
    evaluation_df_to_latex_table(df, output_path)


if __name__ == "__main__":
    main()
