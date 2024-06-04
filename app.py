import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("./heart_cleveland_upload.csv")

df = load_data()

# Sidebar for model selection and user input
st.sidebar.title("Heart Disease Analysis")
st.sidebar.subheader("Choose Model")
model_name = st.sidebar.selectbox("Model", ["Logistic Regression", "LDA", "KNN", "Decision Tree", "Gaussian Naive Bayes", "Random Forest", "SVC"])

st.sidebar.subheader("Input Features for Prediction")
input_data = {
    "age": st.sidebar.slider("Age", min_value=0, max_value=100, value=50),
    "sex": st.sidebar.selectbox("Sex", [0, 1]),
    "trestbps": st.sidebar.slider("Resting Blood Pressure", min_value=0, max_value=200, value=120),
    "chol": st.sidebar.slider("Cholesterol", min_value=0, max_value=600, value=200),
    "fbs": st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1]),
    "restecg": st.sidebar.selectbox("Resting ECG", [0, 1, 2]),
    "thalach": st.sidebar.slider("Max Heart Rate", min_value=0, max_value=250, value=150),
    "exang": st.sidebar.selectbox("Exercise-Induced Angina", [0, 1]),
    "oldpeak": st.sidebar.slider("ST Depression", min_value=0.0, max_value=10.0, value=1.0),
    "ca": st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3]),
    "cp_0": st.sidebar.selectbox("Chest Pain Type 0", [0, 1]),
    "cp_1": st.sidebar.selectbox("Chest Pain Type 1", [0, 1]),
    "cp_2": st.sidebar.selectbox("Chest Pain Type 2", [0, 1]),
    "cp_3": st.sidebar.selectbox("Chest Pain Type 3", [0, 1]),
    "thal_0": st.sidebar.selectbox("Thalassemia Type 0", [0, 1]),
    "thal_1": st.sidebar.selectbox("Thalassemia Type 1", [0, 1]),
    "thal_2": st.sidebar.selectbox("Thalassemia Type 2", [0, 1]),
    "thal_3": st.sidebar.selectbox("Thalassemia Type 3", [0, 1]),
    "slope_0": st.sidebar.selectbox("Slope Type 0", [0, 1]),
    "slope_1": st.sidebar.selectbox("Slope Type 1", [0, 1]),
    "slope_2": st.sidebar.selectbox("Slope Type 2", [0, 1])
}

# Encode categorical features and prepare the data
df[["cp", "thal", "slope"]] = df[["cp", "thal", "slope"]].astype(int)
df_encoded = pd.get_dummies(df, columns=["cp", "thal", "slope"], prefix_sep='_', dtype=int)
x = df_encoded.drop('condition', axis=1)
y = df_encoded['condition']
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=4)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "LDA": LinearDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_features='sqrt'),
    "SVC": SVC()
}

model = models[model_name]
model.fit(X_train, Y_train)

# Evaluate the model using cross-validation
cv_results = cross_val_score(model, X_train, Y_train, cv=10)
mean_score = round(cv_results.mean(), 4)

# Display evaluation metrics
st.title("Heart Disease Data Analysis")
st.subheader(f"{model_name} Model Evaluation")
st.write(f"Mean accuracy from cross-validation: {mean_score}")

# Prepare input data for prediction
input_df = pd.DataFrame([input_data])
missing_cols = set(df_encoded.columns) - set(input_df.columns) - {'condition'}
for col in missing_cols:
    input_df[col] = 0
input_df = input_df[df_encoded.drop(columns='condition').columns]
input_df = scaler.transform(input_df)

# Predict the condition based on user input
prediction = model.predict(input_df)[0]
condition = "Diseased" if prediction == 1 else "Not Diseased"

st.sidebar.subheader("Prediction")
st.sidebar.write(f"The predicted condition is: **{condition}**")

# Data visualization section
st.subheader("Data Visualization")

# Heatmap for correlations
st.subheader("Feature Correlations")
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'condition']
selected_columns = df[numeric_features]
corr_data = selected_columns.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_data, annot=True, cmap='RdBu', linewidths=0.1)
st.pyplot(plt)

# Health condition distribution
st.subheader("Distribution of Health Conditions")
plt.figure()
sns.countplot(x=df["condition"], palette='bwr')
st.pyplot(plt)

# Add more visualizations as needed...
# For example, visualizing distributions of various features and their relationships with the condition
