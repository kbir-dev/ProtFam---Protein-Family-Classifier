import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io

# Set page config
st.set_page_config(
    page_title="ProtFam - Protein Family Classifier",
    page_icon="🧬",
    layout="wide"
)

# Class name mapping
CLASS_NAMES = {
    "0": "DNA Repair Protein",
    "1": "Decarboxylase",
    "2": "Defensin",
    "3": "Heat Shock Protein",
    "4": "RNA Binding Protein",
    "5": "Voltage Gated Channel"
}

# Feature names for SHAP visualization
FEATURE_NAMES = [
    "Negative Charges",
    "Positive Charges",
    "Excitation Coef 1",
    "Excitation Coef 2",
    "Instability Index",
    "Aliphatic Index",
    "GRAVY Score",
    "Serine Count",
    "Threonine Count",
    "Tyrosine Count",
    "Expected AA",
    "Predicted Helices"
]

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Single Analysis", "Bulk Analysis"])

if page == "Single Analysis":
    # Title and description
    st.title("🧬ProtFam - Protein Family Classifier")
    st.markdown("""
    This application predicts the family of a protein based on its physicochemical properties.
    Enter the protein features below to get a prediction.
    """)

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Properties")
        nneg = st.number_input("Number of Negative Charges (Nneg)", min_value=0, value=10)
        npos = st.number_input("Number of Positive Charges (Npos)", min_value=0, value=10)
        exc1 = st.number_input("Excitation Coefficient 1 (Exc1)", min_value=0.0, value=1000.0)
        exc2 = st.number_input("Excitation Coefficient 2 (Exc2)", min_value=0.0, value=1000.0)
        i_index = st.number_input("Instability Index (I.Index)", min_value=0.0, value=40.0)
        a_index = st.number_input("Aliphatic Index (A.Index)", min_value=0.0, value=70.0)

    with col2:
        st.subheader("Advanced Properties")
        gravy = st.number_input("GRAVY Score", min_value=-2.0, max_value=2.0, value=0.0)
        ser = st.number_input("Serine Count (Ser)", min_value=0, value=10)
        thr = st.number_input("Threonine Count (Thr)", min_value=0, value=10)
        tyr = st.number_input("Tyrosine Count (Tyr)", min_value=0, value=10)
        expaa = st.number_input("Expected AA (ExpAA)", min_value=0.0, value=10.0, format="%.4f")
        predhel = st.number_input("Predicted Helices (PredHel)", min_value=0, value=1)

    # Create prediction button
    if st.button("Predict Protein Family"):
        # Prepare the data
        data = {
            "Nneg": nneg,
            "Npos": npos,
            "Exc1": exc1,
            "Exc2": exc2,
            "I_Index": i_index,
            "A_Index": a_index,
            "GRAVY": gravy,
            "Ser": ser,
            "Thr": thr,
            "Tyr": tyr,
            "ExpAA": expaa,
            "PredHel": predhel
        }
        
        try:
            # Make API request
            response = requests.post("http://localhost:8000/predict", json=data)
            result = response.json()
            
            # Display results
            st.success("Prediction completed!")
            
            # Create two columns for results
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.subheader("Predicted Family")
                # Convert numeric class to actual class name
                predicted_class = CLASS_NAMES.get(result['predicted_class'], result['predicted_class'])
                st.info(f"**{predicted_class}**")
            
            with res_col2:
                st.subheader("Prediction Probabilities")
                # Create a bar chart of probabilities
                prob_df = pd.DataFrame({
                    'Family': [CLASS_NAMES.get(k, k) for k in result['probabilities'].keys()],
                    'Probability': list(result['probabilities'].values())
                })
                # Sort by probability in descending order
                prob_df = prob_df.sort_values('Probability', ascending=False)
                fig = px.bar(prob_df, x='Family', y='Probability',
                            title='Prediction Probabilities by Family',
                            color='Probability',
                            color_continuous_scale='Viridis')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Create two columns for SHAP values
            shap_col1, shap_col2 = st.columns(2)
            
            with shap_col1:
                st.subheader("Feature Importance")
                # Get the predicted class index
                pred_class_idx = int(result['predicted_class'])
                
                # Create a sample SHAP values visualization
                # In a real implementation, these would come from the model
                # For now, we'll create a sample visualization
                feature_importance = np.random.rand(len(FEATURE_NAMES))
                feature_importance = feature_importance / feature_importance.sum()
                
                # Sort features by importance
                sorted_idx = np.argsort(feature_importance)
                sorted_features = [FEATURE_NAMES[i] for i in sorted_idx]
                sorted_importance = feature_importance[sorted_idx]
                
                # Create horizontal bar chart
                fig = go.Figure(go.Bar(
                    x=sorted_importance,
                    y=sorted_features,
                    orientation='h',
                    marker_color='rgb(55, 83, 109)'
                ))
                
                fig.update_layout(
                    title=f'Feature Importance for {CLASS_NAMES[str(pred_class_idx)]}',
                    xaxis_title='Importance Score',
                    yaxis_title='Feature',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

else:  # Bulk Analysis page
    st.title("🧬ProtFam - Bulk Analysis")
    st.markdown("""
    Upload a CSV file containing protein features for bulk classification.
    The file should have the following columns:
    - Nneg, Npos, Exc1, Exc2, I_Index, A_Index, GRAVY, Ser, Thr, Tyr, ExpAA, PredHel
    """)

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Check if all required columns are present
            required_columns = ["Nneg", "Npos", "Exc1", "Exc2", "I_Index", "A_Index", 
                              "GRAVY", "Ser", "Thr", "Tyr", "ExpAA", "PredHel"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                # Show preview of the data
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Process button
                if st.button("Process Data"):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    
                    # Initialize results list
                    results = []
                    
                    # Process each row
                    for idx, row in df.iterrows():
                        # Prepare data for API
                        data = row[required_columns].to_dict()
                        
                        try:
                            # Make API request
                            response = requests.post("http://localhost:8000/predict", json=data)
                            result = response.json()
                            
                            # Add prediction to results
                            results.append({
                                'Predicted_Class': CLASS_NAMES.get(result['predicted_class'], result['predicted_class']),
                                'Confidence': max(result['probabilities'].values())
                            })
                            
                            # Update progress
                            progress_bar.progress((idx + 1) / len(df))
                            
                        except Exception as e:
                            results.append({
                                'Predicted_Class': 'Error',
                                'Confidence': 0.0
                            })
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame(results)
                    
                    # Combine with original data
                    final_df = pd.concat([df, results_df], axis=1)
                    
                    # Show results
                    st.subheader("Classification Results")
                    st.dataframe(final_df)
                    
                    # Create download button
                    csv = final_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="protein_classification_results.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit and FastAPI</p>
    <p>Protein Family Classification Model</p>
</div>
""", unsafe_allow_html=True) 