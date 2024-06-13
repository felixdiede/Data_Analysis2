import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy import stats
from scipy.stats import entropy, wasserstein_distance, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


"""
Chapter 1: Univariate Resemblance Analysis
    1.1 Statistical test for numerical attributes
        1.1.1 Student T-test for the comparison of means
        1.1.2 Mann Whitney U-test for population comparison
        1.1.3 Kolmogorov-Smirnov test for distribution comparison
"""
def t_test(real_data, synthetic_data, attribute, alpha=0.05):
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])

    t_statistic, p_value = stats.ttest_ind(vector1, vector2, alternative="two-sided")

    if p_value < alpha:
        conclusion = "H0 is rejected. There is an significant statistical difference between the groups."
    else:
        conclusion = "H0 is not denied. There is no evidence of significant statistical difference between the groups."

    return t_statistic, p_value, conclusion



def print_results_t_test(real_data, synthetic_data, attribute):
    results = {}
    for attr in attribute:
        t_statistic, p_value, conclusion = t_test(real_data, synthetic_data, attr)

        results[attr] = {
            "t-Statistic": t_statistic,
            "p-Value": p_value,
            "Conclusion": conclusion
        }

    results_df = pd.DataFrame(results).transpose()
    print(results_df.to_markdown(numalign="left", stralign="left"))



def mw_test(real_data, synthetic_data, attribute, alpha=0.05):
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])

    mw_statistic, p_value = stats.mannwhitneyu(vector1, vector2, alternative="two-sided")

    if p_value < alpha:
        conclusion = "H0 is rejected. The distributions of the samples are statistically significantly different."
    else:
        conclusion = "H0 is not rejected. The distributions of the samples are statistically not significantly different."

    return mw_statistic, p_value, conclusion



def print_results_mw_test(real_data, synthetic_data, attribute):
    results = {}
    for attr in attribute:
        mw_statistic, p_value, conclusion = mw_test(real_data, synthetic_data, attr)

        results[attr] = {
            "mw-Statistic": mw_statistic,
            "p-Value": p_value,
            "Conclusion": conclusion
        }

    results_df = pd.DataFrame(results).transpose()
    print(results_df.to_markdown(numalign="left", stralign="left"))



def ks_test(real_data, synthetic_data, attribute, alpha=0.05):
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])

    ks_statistic, p_value = stats.ks_2samp(vector1, vector2, alternative = "two-sided")

    if p_value < alpha:
        conclusion = "H0 is rejected. Distributions are not the same."
    else:
        conclusion = "H0 is not rejected. Distributions are the same."

    return ks_statistic, p_value, conclusion



def print_results_ks_test(real_data, synthetic_data, attribute):
    results = {}
    for attr in attribute:
        ks_statistic, p_value, conclusion = ks_test(real_data, synthetic_data, attr)

        results[attr] = {
            "ks-Statistic": ks_statistic,
            "p-Value": p_value,
            "Conclusion": conclusion
        }

    results_df = pd.DataFrame(results).transpose()
    print(results_df.to_markdown(numalign="left", stralign="left"))

"""
Chapter 1: Univariate Resemblance Analysis
    1.2 Statistical test for categorical attributes
        1.2.1 Chi-square test 
"""
def chi2_test(real_data, synthetic_data, attribute, alpha=0.05):
    contingency_table = pd.crosstab(real_data[attribute], synthetic_data[attribute])

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    if chi2 < alpha:
        conclusion = "H0 is rejected."
    else:
        conclusion = "H0 is not rejected."

    return chi2, p, dof, expected, conclusion



def print_results_chi2_test(real_data, synthetic_data, attribute):
    results = {}
    for attr in attribute:
        chi_statistic, p_value, _, _, conclusion = chi2_test(real_data, synthetic_data, attr)

        results[attr] = {
            "chi2-Statistic": chi_statistic,
            "p-Value": p_value,
            "Conclusion": conclusion
        }

    results_df = pd.DataFrame(results).transpose()
    print(results_df.to_markdown(numalign="left", stralign="left"))

"""
Chapter 1: Univariate Resemblance Analysis
    1.3 Distance calculation
        1.3.1 Cosine distance
        1.3.2 Jensen-Shannon Distance
        1.3.3 Kullback-Leibler Divergence
        1.3.4 Wassertein Distance
"""
def cos_distance(real_data, synthetic_data, attribute):
    vector1 = np.array([real_data[attribute]])
    vector2 = np.array([synthetic_data[attribute]])

    cos_dist = cosine_distances(vector1, vector2)

    return cos_dist



def print_cos_distance(real_dat, synthetic_data, attribute):
    results={}
    for attr in attribute:
        cos_distance = cos_distance(real_data, synthetic_data, attr)

        results[attr] = {
            "Cosine distance": cos_distance
        }

    results_df = pd.DataFrame(results).transpose()
    print(results_df.to_markdown(numalign="left", stralign="left"))



def js_distance(real_data, synthetic_data, attribute):
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])
    js_dist = jensenshannon(vector1, vector2)

    return js_dist


def print_js_distance(real_data, synthetic_data, attribute):
    results = {}
    for attr in attribute:
        js_distance = js_distance(real_data, synthetic_data, attr)

        results[attr] = {
            "Josine distance": js_distance
        }

    results_df = pd.DataFrame(results).transpose()
    print(results_df.to_markdown(numalign="left", stralign="left"))



def kl_divergence(real_data, synthetic_data, attribute):
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])

    p = np.histogram(vector1)[0] / len(vector1)
    q = np.histogram(vector2)[0] / len(vector2)

    kl_pq = entropy(p, q)

    return kl_pq


def print_kl_divergence(real_data, synthetic_data, attribute):
    results = {}
    for attr in attribute:
        kl_div = kl_divergence(real_data, synthetic_data, attr)

        results[attr] = {
            "KS-Divergence": kl_div
        }

    results_df = pd.DataFrame(results).transpose()
    print(results_df.to_markdown(numalign="left", stralign="left"))


def was_distance(real_data, synthetic_data, attribute):
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])

    u_values = np.histogram(vector1)[0] / len(vector1)
    v_values = np.histogram(vector2)[0] / len(vector2)

    ws_distance = wasserstein_distance(u_values, v_values)

    return ws_distance

def calculate_and_display_distances(real_data, synthetic_data, attribute):

    # Dictionary with threshold values for each metric
    thresholds = {
        "Cosinus": 0.3,
        "Jensen-Shannon": 0.1,
        "KL-Divergenz": 0.1,
        "Wasserstein": 0.3
    }

    distance_functions = {
        "Cosinus": cos_distance,
        "Jensen-Shannon": js_distance,
        "KL-Divergenz": kl_divergence,
        "Wasserstein": was_distance
    }

    all_results = {}
    for distance_name, distance_func in distance_functions.items():
        results = []
        for attr in attribute:
            try:
                distance = distance_func(real_data, synthetic_data, attr)
                results.append({"Attribut": attr, "Distanz": distance})
            except ValueError:
                results.append({"Attribut": attr, "Distanz": "N/A (Error)"})

        df = pd.DataFrame(results)

        # Add Conclusion column
        df["Conclusion"] = df["Distanz"].apply(
            lambda x: "true" if x < thresholds[distance_name] else "false"
        )

        markdown_table = df.to_markdown(index=False, numalign="left", stralign="left")
        all_results[distance_name] = markdown_table

    for distance_name, markdown_table in all_results.items():
        print(f"\n## {distance_name} Distanzen\n")
        print(markdown_table)




"""
Chapter 2: Multivariate Relationship Analysis
    2.1 PPC Matrices comparison
"""

def ppc_matrix(real_data, synthetic_data, num_features):

    num_real_data = real_data[num_features]
    num_synthetic_data = synthetic_data[num_features]

    corr_matrix_real = num_real_data.corr()
    corr_matrix_syn = num_synthetic_data.corr()

    diff_matrix = corr_matrix_real - corr_matrix_syn

    plt.figure(figsize=(10, 8))
    sns.heatmap(diff_matrix, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title('Differenz der Korrelationsmatrizen')
    plt.show()

"""
Chapter 2: Multivariate Relationship Analysis
    2.2 Normalized contingency tables comparison
"""
def normalized_contingency_tables(real_data, synthetic_data, attributes):
    results = {}
    for attr in attributes:
        table = pd.crosstab(real_data[attr], synthetic_data[attr], normalize='all')
        absolute_deviation = np.sum(np.abs(table - np.outer(table.sum(axis=1), table.sum(axis=0))))

        results[attr] = {
            "Contingency tables": table,
            "Absolute deviation": absolute_deviation
        }
    return results

"""
Chapter 3: DLA 
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def data_labelling_analysis(real_data, synthetic_data):
    # Label real data with 0 and synthetic data with 1
    real_data["label"] = 0
    synthetic_data["label"] = 1

    # Merge both dataframes
    df = pd.concat(real_data, synthetic_data)

    # Create a feature dataset and a target dataset
    X = df.drop("label")
    y = df["label"]

    # Split the data into train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing useless?

    # Create classifier
    model1 = RandomForestClassifier(n_estimators = 100, n_jobs = 3, random_state = 9)
    model2 = KNeighborsClassifier(n_neighbors =10, n_jobs = 3)
    model3 = DecisionTreeClassifier(random_state = 9)
    model4 = SVC(C = 100, max_iter = 300, kernel = "linear", probability = True, random_state = 9)
    model5 = MLPClassifier(hidden_layer_sizes = (128,64,32), max_iteration = 300, random_state = 9)

    classifier = ["model1", "model2", "model3", "model4", "model5"]

    # Train on train dataset
    for model in classifier:
        model.fit(X_train, y_train)

    for model in classifier:
        print(2)








# def data_labelling_analysis(real_data, synthetic_data, classifiers=None):
    test_size = 0.2

    # Label real and synthetic data
    real_data["label"] = 0
    synthetic_data["label"] = 1
    combined_data = pd.concat([real_data, synthetic_data])

    X = combined_data.drop("label", axis=1)
    y = combined_data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Separate numerical and categorical features
    num_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X_train.select_dtypes(include=["object", "category"]).columns

    # Standardize numerical features
    scaler = StandardScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_encoded = encoder.fit_transform(X_train[cat_features])
    X_test_encoded = encoder.transform(X_test[cat_features])

    # Combine features
    X_train_final = pd.concat([X_train[num_features], pd.DataFrame(X_train_encoded.toarray(),
                                                                   columns=encoder.get_feature_names_out(
                                                                       cat_features))], axis=1)
    X_test_final = pd.concat([X_test[num_features], pd.DataFrame(X_test_encoded.toarray(),
                                                                 columns=encoder.get_feature_names_out(cat_features))],
                             axis=1)

    # Default classifiers if none are provided
    if classifiers is None:
        classifiers = [
            RandomForestClassifier(),
            KNeighborsClassifier(),
            DecisionTreeClassifier(),
            SVC(),
            MLPClassifier()
        ]

    # Evaluate each classifier
    results = {}
    for clf in classifiers:
        clf.fit(X_train_final, y_train)
        y_pred = clf.predict(X_test_final)
        results[clf.__class__.__name__] = classification_report(y_test, y_pred, output_dict=True)

    return results



