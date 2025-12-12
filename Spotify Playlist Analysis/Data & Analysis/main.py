# from src.data_prep import run_eda_and_merge
# from src.clustering import run_clustering
# from src.splitting import run_split_and_create_template
from src.modeling import run_supervised_modeling


def main() -> None:
    # run_eda_and_merge()
    # run_clustering()
    # run_split_and_create_template()

    run_supervised_modeling()


if __name__ == "__main__":
    main()
