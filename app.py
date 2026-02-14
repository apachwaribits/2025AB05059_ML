import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Student Dropout Prediction")
st.title("Student Dropout & Academic Success Predictor")

# a. Dataset upload option (CSV) [cite: 91]
uploaded_file = st.file_uploader("Upload Test Data (CSV)", type="csv")

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    st.write("Test Data Preview:", test_df.head())

    # b. Model selection dropdown [cite: 92]
    model_name = st.selectbox("Select Model", 
        ["Logistic_Regression", "Decision_Tree", "kNN", "Naive_Bayes", "Random_Forest", "XGBoost"])

    # Load Model
    with open(f'model/{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)

    if st.button("Evaluate"):
        # Split features and target (assuming 'Target' is last column)
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]
        
        preds = model.predict(X_test)
        
        # c. Display metrics [cite: 93]
        st.subheader(f"Results for {model_name}")
        st.metric("Model Accuracy", f"{accuracy_score(y_test, preds):.4f}")
        
        # d. Confusion matrix [cite: 94]
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Greens')
        st.pyplot(fig)
        
        st.subheader("Classification Report")
        st.text(classification_report(y_test, preds))