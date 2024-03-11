import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.sparse import csr_matrix
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Starting constants and configurations
DATA_FILE = 'AFL-2022-totals.csv'
FULL_TRAIT_NAMES = {
    "Tm": "Team", "GM": "Games played",
    "KI": "Kicks", "MK": "Marks", 
    "HB": "Handballs", "DI": "Disposals", 
    "GL": "Goals", "BH": "Behinds", 
    "HO": "Hit outs", "TK": "Tackles", 
    "RB": "Rebound SOs", "IF": "Inside SOs", 
    "CL": "Clearances", "CG": "Clangers", 
    "FF": "Free kicks for", "FA": "Free kicks against", 
    "BR": "Brownlow votes", "CP": "Contested possessions", 
    "UP": "Uncontested possessions", "CM": "Contested marks", 
    "MI": "Marks inside SO", "1%": "One percenters", 
    "BO": "Bounces", "GA": "Goal assist"
    }

# Note: we were able to create the best results using these constants
N_NEIGHBOURS = 5
N_FOLDS = 5
N_BOOSTRAP_SAMPLES = 100
MAX_DEPTH_REGRESSION_TREE = 6

# Finds the values that are most correlated with goals (pearsons correlation coefficient)
def calculate_pearson_correlation(df, trait):
    """Calculate the Pearson's correlation coefficient between a trait and goals"""
    correlation = df.dropna(subset=[trait, "GL"])
    pearsons_corr_coef = correlation["GL"].corr(correlation[trait])
    return pearsons_corr_coef

# Finds the values that are most correlated with goals (spearmans correlation coefficient (can be non-linear))
def calculate_spearmans_correlation(df, trait):
    """Calculate the Spearman's correlation coefficient between a trait and goals"""
    correlation = df.dropna(subset=[trait, "GL"])
    rho, p_value = spearmanr(correlation[trait], correlation["GL"])
    # p_values of relevant traits were all small so we can just return rho
    # as a means to support the pearson evalutaion of correlation
    return rho

def plot_scatter(df, trait_to_compare, fig_num):
    """Plot a simple scatterplot with a regression line between a trait and goals"""
    # get axis values
    df_axis_vals = df.dropna(subset=[trait_to_compare, "GL"])

    # scatter
    plt.scatter(x=df_axis_vals[trait_to_compare], y=df_axis_vals.loc[:,"GL"])

    # regression line
    x = df_axis_vals[trait_to_compare].values.reshape(-1, 1)
    y = df_axis_vals.loc[:,"GL"].values.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    m = model.coef_[0][0]
    b = model.intercept_[0]

    # plot regression line onto plot
    plt.plot(df_axis_vals[trait_to_compare], (m * df_axis_vals[trait_to_compare] + b), color="orange")

    plt.ylabel(f'Goals')
    plt.xlabel(f'{FULL_TRAIT_NAMES[trait_to_compare]}')
    plt.title(f'Figure {fig_num}: Goals vs {FULL_TRAIT_NAMES[trait_to_compare]}')
    
    plt.savefig(f'correlation_graphs/Goals vs {FULL_TRAIT_NAMES[trait_to_compare]}')

    plt.clf()

def cross_validation_evaluate_knn_model(X, y):
    """Takes in traits(X) and goals(y) creates KNN model and evaluates using k-fold 
    cross-validation"""
    # impute data, change all NaN to 0
    X = X.fillna(0)
    y = y.fillna(0)

    #changing X into a dense matrix
    X = csr_matrix(X).todense()

    # apply n-fold cross validation onto our dataframe
    nf_CV = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    results = []

    for train_idx, test_idx in nf_CV.split(X):
        # train-test split
        X_train, X_test = np.asarray(X[train_idx]), np.asarray(X[test_idx])
        y_train, y_test = np.asarray(y[train_idx]), np.asarray(y[test_idx])
        
        # Preprocessing
        # 1. Standardise the data
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Training
        knn = KNN(n_neighbors=N_NEIGHBOURS)
        knn.fit(X_train, y_train)    
        
        # Predictions
        y_pred = knn.predict(X_test)
        results.append(accuracy_score(y_test, y_pred))

    print(f"Average accuracy evaluated using k-fold cross validation: {np.mean(results)}\n")
 
def bootstrapping_evaluate_knn_model(X, y):
    """Takes in traits(X) and goals(y) creates KNN model and evaluates using boostrapping"""
    # impute data, change all NaN to 0
    X = X.fillna(0)
    y = y.fillna(0)

    X = csr_matrix(X).todense()

    n = X.shape[0]
    dataidx = range(n)

    accuracies = []

    for j in range(N_BOOSTRAP_SAMPLES):
        boot_index = resample(range(n), replace=True, n_samples=n, random_state=j)

        oob_index = [x for x in range(n) if x not in boot_index]
        
        X_imb_train = np.asarray(X[boot_index,:])
        X_imb_test = np.asarray(X[oob_index,:])
        y_imb_train = np.asarray(y[boot_index])
        y_imb_test = np.asarray(y[oob_index])

        knn = KNN(n_neighbors=N_NEIGHBOURS)
        knn.fit(X_imb_train, y_imb_train)
        y_imb_pred=knn.predict(X_imb_test)
        accuracies.append(accuracy_score(y_imb_test, y_imb_pred))

    # Display average of accuracy scores
    avg_acc_score = np.mean(accuracies)
    print(f'Average accuracy evaluated using bootstrapping: {avg_acc_score}\n')

# Create linear regression model; evalute using MSE and accuracy_score
def evaluate_linear_regression(X, y):
    """Takes in traits(X) and goals(y) creates linear regression model and evaluates using mse
    and accuracy_score"""
    # impute data, change all NaN to 0
    X = X.fillna(0)
    y = y.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # evaluate and print evalution scores
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean squared error from linear regression: {mse}')
    score = model.score(X_test, y_test)
    print(f'Test set accuracy from linear regression: {score:.3f}\n')

# Create model using regression tree; evalute using MSE
def evaluate_regression_tree(X, y):
    """Evaluate and plot the regression tree model"""
    # impute data, change all NaN to 0
    X = X.fillna(0)
    y = y.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg_tree = DecisionTreeRegressor(max_depth=MAX_DEPTH_REGRESSION_TREE)

    reg_tree.fit(X_train, y_train)

    y_pred = reg_tree.predict(X_test)

    # evaluate and print evalution scores
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean squared error from regression tree: {mse}')

    # Create plot
    plt.figure(figsize=(85, 20))
    plot_tree(reg_tree, feature_names=['MI', 'BH', 'CM', 'GA', 'GM'], filled=True,
            fontsize=10)
    plt.tight_layout()
    plt.savefig("regression tree")

# LOAD AFL DATA FROM CSV FILE
df = pd.read_csv('AFL-2022-totals.csv')

# Not looking at team/player just looking at the numeric data
traits = list(df.columns.values)
traits.remove("Player")
traits.remove("TM")

# print all the results
ranked_pearsons_corr_coef = {}   
ranked_spearmans_corr_coef = {}
# print the correlation coefficient strengths for pearsons and spearmans
for trait in traits:
    pearsons_corr_coef = calculate_pearson_correlation(df, trait)
    ranked_pearsons_corr_coef[trait] = pearsons_corr_coef

    spearmans_corr_coef = calculate_spearmans_correlation(df, trait)
    ranked_spearmans_corr_coef[trait] = spearmans_corr_coef

# sort and print the dictionaries of correlation coefficients from highest to lowest values
ranked_pearsons_corr_coef = {k: v for k,v in sorted(ranked_pearsons_corr_coef.items(), key=lambda item: item[1], reverse=True)}
print(f'Ranked pearsons correlation coefficients: {ranked_pearsons_corr_coef}\n')
ranked_spearmans_corr_coef = {k: v for k,v in sorted(ranked_spearmans_corr_coef.items(), key=lambda item: item[1], reverse=True)}
print(f'Ranked spearmans correlation coefficients: {ranked_spearmans_corr_coef}\n')

# get all simple scatterplots with regression lines; name the figures
fig_num = 1
for trait in traits:
    plot_scatter(df, trait, fig_num)
    fig_num += 1

# from testing |pearsons coef| >= 0.4 is what we based our choices on

# if you want to test different contributing factors feel free to change the parameters :)
# the code below will let you play around with how many of the most weighted elements
# you want to include in the parameters; you will have to change some of the code in regression
# tree but the other models will run

# best_traits = list(ranked_pearsons_corr_coef)
# best_traits.pop(0)
# X = df[best_traits[:5]]

X = df[['MI', 'BH', 'CM', 'GA', 'GM']]
y = df['GL']

# create knn model and evalute using cross-validation
cross_validation_evaluate_knn_model(X, y)

# create knn model and evalute using bootstrapping
bootstrapping_evaluate_knn_model(X, y)

# create linear regression model and evalute mse and accuracy_score
evaluate_linear_regression(X, y)

# create regression and evalute using mse
evaluate_regression_tree(X, y)


