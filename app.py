import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Student Dropout Prediction")
st.title("🎓 Student Dropout & Academic Success Predictor")

# a. Dataset upload option (CSV) [1 mark]
uploaded_file = st.file_uploader("Upload Test Data (CSV)", type="csv")

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    st.write("### Test Data Preview", test_df.head())

    # b. Model selection dropdown [1 mark]
    model_options = ["Logistic_Regression", "Decision_Tree", "kNN", "Naive_Bayes", "Random_Forest", "XGBoost"]
    model_name = st.selectbox("Select Model for Evaluation", model_options)

    # Load Model
    try:
        with open(f'model/{model_name}.pkl', 'rb') as f:
            model = pickle.load(f)
        
        if st.button("Run Evaluation"):
            # Split features and target
            # Assuming 'Target' is the column name based on your UCI metadata
            if 'Target' in test_df.columns:
                X_test = test_df.drop(columns=['Target'])
                y_test = test_df['Target']
            else:
                X_test = test_df.iloc[:, :-1]
                y_test = test_df.iloc[:, -1]

            # IMPORTANT: Re-apply Scaling
            # In a real app, you should also pickle the scaler used during training.
            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(X_test)
            
            # Predict
            preds = model.predict(X_test_scaled)
            
            # c. Display evaluation metrics [1 mark]
            st.divider()
            col1, col2 = st.columns(2)
            acc = accuracy_score(y_test, preds)
            col1.metric("Model Accuracy", f"{acc:.4f}")
            col2.write("Evaluation complete.")

            # d. Confusion matrix or classification report [1 mark]
            st.subheader("Visualizing Performance")
            
            tab1, tab2 = st.tabs(["Confusion Matrix", "Classification Report"])
            
            with tab1:
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues')
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                st.pyplot(fig)
            
            with tab2:
                st.text("Detailed Metrics:")
                st.text(classification_report(y_test, preds))
                
    except FileNotFoundError:
        st.error(f"Model file '{model_name}.pkl' not found in the 'model/' directory. Please ensure your models are trained and saved.")
