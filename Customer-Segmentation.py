import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# Use st.cache_data to cache data loading functions
@st.cache_data
def data_load():
    customer_data = pd.read_csv('static/Customers-segmentation-dataset.csv')
    return customer_data

# Load data
data = data_load()

st.markdown("# TVS Credit E.P.I.C 6.0 -- IT Challenge.")
st.header("1. Customer Classification for Personalized Recommendations.")

# Display Dataset Overview
st.write("Dataset contains the following information")
st.write("""
- **Gender**: Can be predicted using name or accessed directly.
- **Age**: Calculated from the date of birth.
- **Annual Income**: Indicates earning capacity, which correlates with spending potential.
- **Spending Score**: A measure of spending habits based on the number and value of transactions. It is calculated using [Average money spent x No. of transactions]
""")
st.write("Dataset overview - This dataset contains factors based on which the spending score is calculated.") 
st.write(data)

#----------------------------------------------------------------------------------------------------------------------------------------

st.write("Graphical Overview:")
# Filter out non-numeric data for correlation calculation
numeric_data = data.select_dtypes(include=[np.number])

# Plotting Age Distribution
fig_age = px.histogram(data, x='Age', title='Distribution of Age', nbins=10, color_discrete_sequence=['blue'], marginal='rug')
st.plotly_chart(fig_age)

# Plotting Annual Income Distribution
fig_income = px.histogram(data, x='Annual Income (k$)', title='Annual Income Distribution', nbins=10, color_discrete_sequence=['blue'], marginal='rug')
st.plotly_chart(fig_income)

# Correlation Heatmap
fig_heatmap = px.imshow(numeric_data.corr(), title='Correlation Heatmap', color_continuous_scale='RdBu')
st.plotly_chart(fig_heatmap)
st.write('--------------------')

#----------------------------------------------------------------------------------------------------------------------------------------

# Define cluster labels at the beginning
income_spend_cluster_labels = {0: "Average", 1: "Spenders", 2: "Best", 3: "Low Budget", 4: "Savers"}
age_spend_cluster_labels = {0: "Regular Customers", 1: "Young Targets", 2: "Usual Customers", 3: "Old Targets"}
income_total_cluster_labels = {0: "Rich", 1: "Middle Class", 2: "Poor"}

# Section for Annual Income and Spending Score Classification
st.write('### Classification based on Annual Income and Spending Score.')
# KMeans Clustering on Annual Income and Spending Score
Income_Spend = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

wcss = []
for i in range(1, 13):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=400, n_init=10, random_state=0)
    km.fit(Income_Spend)
    wcss.append(km.inertia_)

# Elbow Graph
fig_elbow = px.line(x=range(1, 13), y=wcss, title='The Elbow Graph', labels={'x': 'Number of Clusters', 'y': 'WCSS'})
st.plotly_chart(fig_elbow)

kmeans_income_spend = KMeans(n_clusters=5, init='k-means++', max_iter=400, n_init=10, random_state=0)
y_means_income_spend = kmeans_income_spend.fit_predict(Income_Spend)

# User Input Section for Income & Spend Classification
st.write("### Classify a New Customer (Income & Spend)")
annual_income_input = st.number_input("Annual Income (k$)", min_value=0, value=50, key='income_input_1')
spending_score_input = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50, key='spending_input_1')

# Button to classify
if st.button("Classify Customer (Income & Spend)"):
    # Prepare input for prediction
    input_data = np.array([[annual_income_input, spending_score_input]])
    predicted_cluster = kmeans_income_spend.predict(input_data)[0]
    
    # Display the predicted cluster
    st.success(f"The customer is classified as: {income_spend_cluster_labels[predicted_cluster]}")
    
    # Create a DataFrame for existing cluster data
    cluster_df_income_spend = pd.DataFrame(Income_Spend, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
    cluster_df_income_spend['Cluster'] = y_means_income_spend

    # Plot existing clusters and the new customer
    fig_prediction_income_spend = px.scatter(cluster_df_income_spend, x='Annual Income (k$)', y='Spending Score (1-100)', 
                                 color='Cluster', title='Customer Classification (Income & Spend)',
                                 color_continuous_scale=px.colors.sequential.Viridis)
    
    # Add the new customer to the plot
    fig_prediction_income_spend.add_scatter(x=[annual_income_input], y=[spending_score_input], mode='markers', 
                                marker=dict(size=15, color='red'), name='New Customer')
    
    st.plotly_chart(fig_prediction_income_spend)

#----------------------------------------------------------------------------------------------------------------------------------------

st.write('---------------------')
# Section for Age and Spending Score Classification
st.write('### Classification based on Age and Spending Score')
# KMeans Clustering on Age and Spending Score
Age_Spend = data[['Age', 'Spending Score (1-100)']].values

wcss = []
for i in range(1, 13):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=400, n_init=10, random_state=0)
    km.fit(Age_Spend)
    wcss.append(km.inertia_)

# Elbow Graph for Age and Spending Score
fig_elbow_age = px.line(x=range(1, 13), y=wcss, title='The Elbow Method for Age and Spending Score', labels={'x': 'Number of Clusters', 'y': 'WCSS'})
st.plotly_chart(fig_elbow_age)

kmeans_age_spend = KMeans(n_clusters=4, init='k-means++', max_iter=400, n_init=10, random_state=0)
y_means_age_spend = kmeans_age_spend.fit_predict(Age_Spend)

# User Input Section for Age & Spend Classification
age_input = st.number_input("Age", min_value=0, max_value=100, value=30, key='age_input_2')
spending_score_input_age = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50, key='spending_input_2')

# Button to classify
if st.button("Classify Customer (Age & Spend)"):
    # Prepare input for prediction
    input_data = np.array([[age_input, spending_score_input_age]])
    predicted_cluster_age = kmeans_age_spend.predict(input_data)[0]
    
    # Display the predicted cluster
    st.success(f"The customer is classified as: {age_spend_cluster_labels[predicted_cluster_age]}")
    
    # Plot existing clusters and the new customer
    fig_age_spend = px.scatter(x=Age_Spend[:, 0], y=Age_Spend[:, 1], color=y_means_age_spend.astype(str), 
                                labels={'x': 'Age', 'y': 'Spending Score (1-100)'},
                                title='Customer Segmentation using Age and Spending Score',
                                color_continuous_scale=px.colors.sequential.Viridis)

    # Add the new customer to the plot
    fig_age_spend.add_scatter(x=[age_input], y=[spending_score_input_age], mode='markers', 
                               marker=dict(size=15, color='red'), name='New Customer')
    
    st.plotly_chart(fig_age_spend)
st.write('--------------------------')

#----------------------------------------------------------------------------------------------------------------------------------------

# Section for Age and Total Income + Spending Score Classification
st.write('### Classification based on Age and Total Income + Spending Score')
# Additional KMeans Clustering Analysis on Total Income and Spending Score
data["Total"] = data["Annual Income (k$)"] + data["Spending Score (1-100)"]
Income_Spend_Total = data[['Age', 'Total']].values

wcss = []
for i in range(1, 13):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=400, n_init=10, random_state=0)
    km.fit(Income_Spend_Total)
    wcss.append(km.inertia_)

# Elbow Graph for Total Income and Spending Score
fig_elbow_total = px.line(x=range(1, 13), y=wcss, title='The Elbow Graph for Age and Total Income + Spending Score', labels={'x': 'Number of Clusters', 'y': 'WCSS'})
st.plotly_chart(fig_elbow_total)

kmeans_total = KMeans(n_clusters=3, init='k-means++', max_iter=400, n_init=10, random_state=0)
y_means_total = kmeans_total.fit_predict(Income_Spend_Total)

# User Input Section for Age & Total (Income + Spend) Classification
age_input_total = st.number_input("Age", min_value=0, max_value=100, value=30, key='age_input_total')
total_income_input = st.number_input("Total Annual Income (k$)", min_value=0, value=50, key='total_income_input')
spending_score_total_input = st.number_input("Total Spending Score", min_value=0, value=50, key='spending_score_total_input')

# Button to classify
if st.button("Classify Customer (Age & Total Income + Spend)"):
    # Prepare input for prediction
    total_input_data = np.array([[age_input_total, total_income_input + spending_score_total_input]])
    predicted_cluster_total = kmeans_total.predict(total_input_data)[0]
    
    # Display the predicted cluster
    st.success(f"The customer is classified as: {income_total_cluster_labels[predicted_cluster_total]}")
    
    # Plot existing clusters and the new customer
    fig_total = px.scatter(x=Income_Spend_Total[:, 0], y=Income_Spend_Total[:, 1], color=y_means_total.astype(str), 
                            labels={'x': 'Age', 'y': 'Total (Income + Spending Score)'},
                            title='Customer Segmentation using Age and Total Income + Spending Score',
                            color_continuous_scale=px.colors.sequential.Viridis)

    # Add the new customer to the plot
    fig_total.add_scatter(x=[age_input_total], y=[total_income_input + spending_score_total_input], mode='markers', 
                           marker=dict(size=15, color='red'), name='New Customer')
    
    st.plotly_chart(fig_total)

st.write('---------------------------')
