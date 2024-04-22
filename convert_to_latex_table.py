from clustpy.utils.evaluation import evaluation_df_to_latex_table


def main():
    df = "C:/Users/Chris/Desktop/Uni/Masterarbeit/Experiment_results/MyBenchmark_Semisupervised_loss/semisupervised_init_small_ff_512_256_128_10/Results/result.csv"
    print(df)
    output_path = df[:-3]+"tex"
    evaluation_df_to_latex_table(df, output_path)


if __name__ == "__main__":
    main()
