import streamlit as st
import requests
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull

# Streamlit UI setup
st.title("Customer Segmentation with KMeans and PCA")

# Upload CSV or Excel file
uploaded_file = st.file_uploader("Upload Customer Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read the file content
        file_content = uploaded_file.read()

        # Send file content to FastAPI for prediction
        files = {"file": ("uploaded_file", io.BytesIO(file_content), "application/octet-stream")}
        response = requests.post("http://127.0.0.1:8000/predict/", files=files)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            df = pd.DataFrame(data)

            # Show customer segmentation data
            st.write("Segmentation Results:")
            st.write(df)

            # Perform PCA on the scaled data
            rfm_scaled = StandardScaler().fit_transform(df[['Recency', 'Frequency', 'Monetary']])
            pca = PCA(n_components=2)
            rfm_pca = pca.fit_transform(rfm_scaled)
            df['PCA1'] = rfm_pca[:, 0]
            df['PCA2'] = rfm_pca[:, 1]

            # Define colors and styles
            plt.figure(figsize=(12, 7))
            sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="viridis", s=80, edgecolor="black", alpha=0.75)

            # Calculate centroids for each cluster
            centroids = df.groupby("Cluster")[["PCA1", "PCA2"]].mean()

            # Plot centroids
            plt.scatter(centroids["PCA1"], centroids["PCA2"], c="red", s=150, marker="X", edgecolors="black", label="Centroids")

            # Optional: Draw Convex Hulls around clusters
            for cluster in df["Cluster"].unique():
                cluster_points = df[df["Cluster"] == cluster][["PCA1", "PCA2"]].values
                if len(cluster_points) > 2:  # Convex hull requires at least 3 points
                    hull = ConvexHull(cluster_points)
                    hull_points = np.append(hull.vertices, hull.vertices[0])  # Close the shape
                    plt.plot(cluster_points[hull_points, 0], cluster_points[hull_points, 1], "--", lw=1.5, color="black", alpha=0.6)

            # Labels and Title
            plt.xlabel("PCA1", fontsize=12)
            plt.ylabel("PCA2", fontsize=12)
            plt.title("Customer Segmentation using PCA", fontsize=14, fontweight="bold")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)

            # Show Plot in Streamlit
            st.pyplot(plt)

        else:
            st.error(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"An error occurred: {e}")