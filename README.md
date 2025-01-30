# Customer Segmentation Dashboard



<div style="display: flex; justify-content: space-between;">
  <img src="Screenshot 2025-01-31 at 2.16.44 AM.png" width="45%" alt="Customer Segmentation Overview">
  <img src="Screenshot 2025-01-31 at 2.17.41 AM.png" width="45%" alt="Customer Segmentation Results">
</div>

## Project Summary:
The Customer Segmentation Dashboard leverages machine learning and data visualization techniques to analyze and segment customers based on transactional data. The dashboard enables users to upload CSV or Excel files containing customer information, such as purchase frequency, recency, and monetary value. These features are processed, and customers are grouped into clusters using the KMeans algorithm. The clusters are then visualized using Principal Component Analysis (PCA) to reduce dimensionality and plot the clusters in a 2D space.

Key Features:
* Customer Data Upload: Users can upload customer data in CSV or Excel format.
* KMeans Clustering: The app segments customers into clusters based on Recency, Frequency, and Monetary (RFM) values using the KMeans clustering algorithm.
* PCA Visualization: The app uses PCA to reduce the dimensionality of the data and plots the customer clusters, with optional convex hulls around each cluster for better visualization.
* Interactive Dashboard: The results, including the segmented data and PCA plot, are displayed in an interactive, easy-to-read format.

Technologies Used:
* Python: For backend and machine learning logic.
* Streamlit: To build an interactive web dashboard.
* FastAPI: To handle predictions and data processing requests.
* Scikit-learn: For machine learning algorithms (KMeans, PCA).
* Matplotlib & Seaborn: For data visualization.

This project showcases the ability to build interactive data dashboards and apply machine learning algorithms to real-world datasets, providing valuable insights into customer behavior


