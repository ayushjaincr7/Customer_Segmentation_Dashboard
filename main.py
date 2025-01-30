from fastapi import FastAPI, UploadFile, File
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import io

app = FastAPI()

# Initialize the KMeans model (or load a pre-trained model)
kmeans = KMeans(n_clusters=4, random_state=42)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file as bytes
    contents = await file.read()

    # Convert the bytes content to a pandas DataFrame (Excel file)
    df = pd.read_excel(io.BytesIO(contents))
    
    df = df.dropna(subset=['CustomerID'])

        # Convert 'InvoiceDate' to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

        # Drop rows with invalid 'InvoiceDate' (NaT)
    df = df.dropna(subset=['InvoiceDate'])

        # Remove negative quantities (assumed returns)
    df = df[df['Quantity'] > 0]

        # Create 'TotalAmount' column
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    # Extract RFM features
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalAmount': 'Monetary'})

    # Standardize the data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # Predict clusters
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Convert DataFrame to JSON and return
    return rfm.to_dict(orient="records")