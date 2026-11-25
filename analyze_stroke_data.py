from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

pd.set_option("display.max_columns", None)

dataset_path = Path(__file__).with_name("healthcare-dataset-stroke-data.csv")
df = pd.read_csv(dataset_path)

print(df.head())
print(df.info())
print(df.describe(include='all'))

df["heart_disease"] = df["heart_disease"].replace({1: "Yes", 0: "No"})

print("*************************************************")
print("****************Corrupted DataFrame**************")
print("*************************************************")

counter=0
for row in df["heart_disease"]:
    try:
        float(row)
        pass
    except ValueError:
        if str(row).strip().upper() == "YES":
            df.loc[counter, "heart_disease"] = 1
        elif str(row).strip().upper() == "NO":
            df.loc[counter, "heart_disease"] = 0
    counter+=1
df["heart_disease"] = df["heart_disease"].astype('float64')

df.drop_duplicates(inplace=True)
df.drop(index=df[df["gender"] == "Other"].index, inplace=True)
df.dropna(inplace=True)

df["gender"] = df["gender"].astype("object")
df["ever_married"] = df["ever_married"].astype("object")
df["work_type"] = df["work_type"].astype("object")
df["Residence_type"] = df["Residence_type"].astype("object")
df["smoking_status"] = df["smoking_status"].astype("object")

df.reset_index(drop=True, inplace=True)

print("*************************************************")
print("****************Cleaned DataFrame****************")
print("*************************************************")

print(df.info())
print(df.describe(include='all'))

print("*************************************************")
print("****************Pivot Tables****************")
print("*************************************************")

pv1 = pd.pivot_table(df, index=["gender", "heart_disease"], columns="smoking_status", values="stroke", aggfunc="mean")
pv1_str = (pv1 * 100).round(2).astype(str) + "%"
print(pv1_str)

pv1.index = ["Female - No heart disease", "Female - Heart disease", "Male - No heart disease", "Male - Heart disease"]
pv1 = (pv1 * 100).round(2)

pv1.plot(kind="bar", figsize=(12, 6), title="Stroke (%) by Gender, Heart Disease and Smoking Status")
plt.xlabel("Gender and Heart Disease Status")
plt.ylabel("Stroke (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

pv1.plot(kind="barh", figsize=(12, 6), title="Stroke (%) by Gender, Heart Disease and Smoking Status")
plt.xlabel("Stroke (%)")
plt.ylabel("Gender and Heart Disease Status")
plt.tight_layout()
plt.show()

pv2 = pd.pivot_table(df, index=["gender", "ever_married"], columns="work_type", values="stroke", aggfunc="mean")
pv2_str = (pv2 * 100).round(2).astype(str) + "%"
print(pv2_str)

pv2.index = ["Female - Not Married", "Female - Married", "Male - Not Married", "Male - Married"]
pv2 = (pv2 * 100).round(2)

pv2.plot(kind="line", stacked=False, figsize=(12, 6), title="Stroke (%) by Gender, Marriage Status and Work Type")
plt.xlabel("Gender and Marriage Status")
plt.ylabel("Stroke (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

pv2.plot(kind="area", stacked=False, figsize=(12, 6), title="Stroke (%) by Gender, Marriage Status and Work Type")
plt.xlabel("Gender and Marriage Status")
plt.ylabel("Stroke (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

pv3 = pd.pivot_table(df, index=["gender", "stroke"], values=["age", "avg_glucose_level", "bmi"], aggfunc="mean")
pv3 = pv3.round(2)
print(pv3)

pv3.index = ["Female - No Stroke", "Female - Stroke", "Male - No Stroke", "Male - Stroke"]

# .annotate() is code from chatGPT to add labels to the scatter plot points.
ax = pv3.plot(kind="scatter", x="age", y="avg_glucose_level", figsize=(12, 6), title="Average Age vs Average Glucose Level")
for group, row in pv3.iterrows():
    ax.annotate(group, (row["age"], row["avg_glucose_level"]), xytext=(5, 5), textcoords="offset points")
plt.xlabel("Average Age")
plt.ylabel("Average Glucose Level")
plt.tight_layout()
plt.show()

pv3.plot(kind="box", figsize=(12, 6), title="Average Age, Glucose Level and BMI")
plt.xlabel("Measurements")
plt.ylabel("Average Values")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sns.violinplot(data=df, x="stroke", y="age", hue="gender", split=True)
plt.title("Age Distribution by Stroke Status and Gender")
plt.xlabel("Stroke Status")
plt.ylabel("Age")
# next line also from chatGPT to add appropriate labels.
plt.xticks([0, 1], ["No Stroke", "Stroke"])
plt.tight_layout()
plt.show()

# next line also from chatGPT to add appropriate labels.
df["stroke_status"] = df["stroke"].map({0: "No Stroke", 1: "Stroke"})
plt.figure(figsize=(12, 6))
sns.ecdfplot(data=df, x="age", hue="stroke_status")
plt.title("Cumulative Age Distribution by Stroke Status")
plt.xlabel("Age")
plt.ylabel("Cumulative Proportion")
plt.tight_layout()
plt.show()

df.plot.hexbin(x="bmi", y="avg_glucose_level", gridsize=20, figsize=(12, 6), title="Density of BMI and Average Glucose Level")
plt.xlabel("BMI")
plt.ylabel("Average Glucose Level")
plt.tight_layout()
plt.show()

X = df.iloc[:, [8, 9]].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# next line from chatGPT to properly show the centroids after normalizing the data.
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c="red", label="Cluster 0")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c="blue", label="Cluster 1")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c="green", label="Cluster 2")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c="cyan", label="Cluster 3")
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c="black", label="Centroids")
plt.title("Clusters of Patients")
plt.xlabel("Average Glucose Level")
plt.ylabel("BMI")
plt.legend()
plt.show()

# next code is from chatGPT to understand the clusters and their relationship with stroke cases.
df["cluster"] = y_kmeans
cluster_summary = df.groupby("cluster").agg(
    patients=("id", "count"),
    stroke_cases=("stroke", "sum"),
    stroke_rate=("stroke", "mean")
)
cluster_summary["stroke_rate"] = (cluster_summary["stroke_rate"] * 100).round(2)
print(cluster_summary)

X = df[["age", "bmi"]]
y = df["avg_glucose_level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Intercept:", round(model.intercept_, 2))
for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, "coefficient:", round(coefficient, 2))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE:", round(root_mean_squared_error(y_test, y_pred), 2))
print("R2:", round(r2_score(y_test, y_pred), 2))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel("Actual Glucose Level")
plt.ylabel("Predicted Glucose Level")
plt.title("Actual vs Predicted Glucose Level")
plt.tight_layout()
plt.show()

# next code is from chatGPT to check if the model and coefficients are statistically significant.
X = df[["age", "bmi"]]
y = df["avg_glucose_level"]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

X = df[["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]]
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# changing tree depth to check best accuracy.
# tree_depth = 3
# tree_depth = 5
tree_depth = 7

gini_decision_tree = DecisionTreeClassifier(max_depth=tree_depth, random_state=42, criterion= "gini")
gini_decision_tree = gini_decision_tree.fit(X_train,y_train)
y_pred = gini_decision_tree.predict(X_test)

print(f"gini decision tree (depth- {tree_depth}) performance:")
print("Accuracy score: ", metrics.accuracy_score(y_test, y_pred))
print("F1 score: ", metrics.f1_score(y_test, y_pred))
print("Recall score: ", metrics.recall_score(y_test, y_pred))
print("Precision score: ", metrics.precision_score(y_test, y_pred))
print("Confusion matrix: ")
print(metrics.confusion_matrix(y_test, y_pred))
preds_prob = gini_decision_tree.predict_proba(X_test)[:, 1]
print("ROC AUC score: ", metrics.roc_auc_score(y_test, preds_prob))

plt.figure(figsize=(50, 25))
plot_tree(gini_decision_tree, filled=True, feature_names=X.columns, class_names=["No Stroke", "Stroke"], rounded=True, fontsize=5, proportion=True, precision=2)
plt.title("Decision tree using gini criterion")
plt.tight_layout
plt.show()

entropy_decision_tree = DecisionTreeClassifier(max_depth=tree_depth, random_state=42, criterion= "entropy")
entropy_decision_tree = entropy_decision_tree.fit(X_train,y_train)
y_pred = entropy_decision_tree.predict(X_test)

print(f"entropy decision tree (depth- {tree_depth}) performance:")
print("Accuracy score: ",metrics.accuracy_score(y_test, y_pred))
print("F1 score: ",metrics.f1_score(y_test, y_pred))
print("Recall score: ",metrics.recall_score(y_test, y_pred))
print("Precision score: ",metrics.precision_score(y_test, y_pred))
print("Confusion matrix: ")
print(metrics.confusion_matrix(y_test, y_pred))
preds_prob = entropy_decision_tree.predict_proba(X_test)[:, 1]
print("ROC AUC score: ", metrics.roc_auc_score(y_test, preds_prob))

plt.figure(figsize=(50, 25))
plot_tree(entropy_decision_tree, filled=True, feature_names=X.columns, class_names=["No Stroke", "Stroke"], rounded=True, fontsize=5, proportion=True, precision=2)
plt.title("Decision tree using entropy criterion")
plt.tight_layout
plt.show()

random_forest = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
random_forest = random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print("\nrandom forest performance")
print("Accuracy score: ", metrics.accuracy_score(y_test, y_pred))
print("F1 score: ", metrics.f1_score(y_test, y_pred))
print("Recall score: ", metrics.recall_score(y_test, y_pred))
print("Precision score: ", metrics.precision_score(y_test, y_pred))
print("Confusion matrix: ")
print(metrics.confusion_matrix(y_test, y_pred))
preds_prob = random_forest.predict_proba(X_test)[:, 1]
print("ROC AUC score: ", metrics.roc_auc_score(y_test, preds_prob))


xgboost = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss")
xgboost = xgboost.fit(X_train, y_train)
y_pred = xgboost.predict(X_test)

print("\nxgboost performance")
print("Accuracy score: ", metrics.accuracy_score(y_test, y_pred))
print("F1 score: ", metrics.f1_score(y_test, y_pred))
print("Recall score: ", metrics.recall_score(y_test, y_pred))
print("Precision score: ", metrics.precision_score(y_test, y_pred))
print("Confusion matrix: ")
print(metrics.confusion_matrix(y_test, y_pred))
preds_prob = xgboost.predict_proba(X_test)[:, 1]
print("ROC AUC score: ", metrics.roc_auc_score(y_test, preds_prob))

# changing hidden layer neuron number to check minimize loss.
hidden_layer_neurons = 8
# hidden_layer_neurons = 16
# hidden_layer_neurons = 32

neural_network = Sequential()
neural_network.add(Dense(hidden_layer_neurons, input_shape=(5,), activation="relu"))
neural_network.add(Dense(1, activation="sigmoid"))
neural_network.compile(loss="binary_crossentropy", optimizer="SGD", metrics=[tf.keras.metrics.AUC()])

neural_network.fit(x=X_train, y=y_train, epochs=10, verbose=2, batch_size=256, validation_data=(X_test, y_test))

nn_losses = pd.DataFrame(neural_network.history.history)
plt.figure(figsize=(15, 5))
sns.lineplot(data=nn_losses, lw=3)
plt.xlabel("Epochs")
plt.ylabel("")
plt.title("Neural Network Training Performance")
sns.despine()
plt.tight_layout()
plt.show()

neural_network.predict(X_test)
nn_y_pred = (neural_network.predict(X_test)>=0.5)[:,0]

print("\nneural network performance: ")
print("Confusion matrix: ")
nn_confusion_matrix = metrics.confusion_matrix(y_test, nn_y_pred)
print(nn_confusion_matrix)

plt.figure(figsize=(8, 4))
sns.set(font_scale=1.4)
ax = sns.heatmap(nn_confusion_matrix, cmap="Reds", annot=True, fmt="d", square=True, xticklabels=["No Stroke (0)", "Stroke (1)"], yticklabels=["No Stroke (0)", "Stroke (1)"])
ax.set(xlabel="Predicted", ylabel="Actual")
plt.title("Neural Network Confusion Matrix")
plt.tight_layout()
plt.show()