from Metrics.Resamblance_Metrics import *

cat_features = []
num_features = ["time", ]

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Results_Analysis/.venv/Data/real/aids_original.csv")
aids_synthetic_tabfairgan = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Results_Analysis/.venv/Data/synthetic/aids_synthetic_tabfairgan.csv")


def statistical_tests(real_data, synthetic_data):
    print_results_t_test(real_data, synthetic_data, attribute=cat_features)
    print_results_mw_test(real_data, synthetic_data, attribute=cat_features)
    print_results_ks_test(real_data, synthetic_data, attribute=cat_features)
    print_results_chi2_test(real_data, synthetic_data, attribute=num_features)

# statistical_tests(diabetes_health_original, diabetes_health_multifairgan)

calculate_and_display_distances(real_data, aids_synthetic_tabfairgan, )

# ppc_matrix(diabetes_health_original, diabetes_health_tabfairgan, num_features)

# normalized_contingency_tables()

# data_labelling_analysis(diabetes_health_original, diabetes_health_tabfairgan)