# Step-by-Step Guide to Run the Enhanced Implementation

## Prerequisites

Before starting, ensure you have:
- **Python 3.7+** installed
- **Java 8+** installed (for Java clients)
- **Web browser** (for web client)
- **Terminal/Command Prompt** access

---

## Step 1: Install Python Dependencies

### On Windows (PowerShell):
```powershell
cd "base paper implementation"
pip install -r requirements.txt
```

### On Linux/Mac:
```bash
cd "base paper implementation"
pip3 install -r requirements.txt
```

**Expected Output**: All packages should install successfully:
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- scikit-learn
- flask
- flask-cors

**If you get errors**, try:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 2: Verify Installation

Test that Python can import all required modules:
```powershell
python -c "import pandas, matplotlib, seaborn, sklearn, flask; print('All dependencies OK')"
```

**Expected Output**: `All dependencies OK`

---

## Step 3: Configure Server (Optional)

The server will work with default settings, but you can customize:

1. Open `config.json` in a text editor
2. Modify settings if needed:
   ```json
   {
       "server": {
           "host": "127.0.0.1",
           "port": 1234,
           "max_connections": 5,
           "timeout_seconds": 30
       }
   }
   ```
3. Save the file

**Note**: Default settings work fine for most cases.

---

## Step 4: Start the Python Server

### Option A: Socket Server (Original)
```powershell
python server.py
```

**Expected Output**:
```
Server listening on 127.0.0.1:1234 ...
Server supports concurrent client connections.
```

**Keep this terminal window open!** The server must stay running.

### Option B: REST API Server (For Web Client)
Open a **new terminal window** and run:
```powershell
cd "base paper implementation"
python api_server.py
```

**Expected Output**:
```
Starting REST API server on http://127.0.0.1:5000
API Documentation:
  GET  /api/health - Health check
  ...
 * Running on http://127.0.0.1:5000
```

**Keep this terminal window open too!**

---

## Step 5: Prepare Your Dataset

1. Place your data file (CSV or TXT) in the `base paper implementation` folder
2. Example: `data.txt` should be in the same folder as `server.py`
3. Format: Comma or space-separated values, one row per line

**Example data.txt format**:
```
2104,3,399900
1600,3,329900
2400,3,369000
```

---

## Step 6: Run Java Client

### Step 6.1: Compile Java Client
Open a **new terminal window**:
```powershell
cd "base paper implementation"
javac JavaSwingClient.java
```

**Expected Output**: No errors (may show warnings, which are OK)

### Step 6.2: Run Java Client
```powershell
java JavaSwingClient
```

**Expected Output**: A window should open with the enhanced Java client interface.

### Step 6.3: Connect to Server
1. In the Java client window, verify:
   - Server: `127.0.0.1`
   - Port: `1234`
2. Click **"Test Connection"** button
3. Status bar should show: **"Connection successful!"** in green

### Step 6.4: Load Dataset
1. Click **"Load File..."** button (or File → Load Dataset...)
2. Navigate to and select `data.txt`
3. Click **Open**
4. Output area should show: `DataFrame 'df' loaded from 'data.txt' with shape (X, Y)`

### Step 6.5: Test Features
Try these quick actions:
- Click **"Summary"** - See statistical summary
- Click **"Shape"** - See dataset dimensions
- Click **"View Data"** - See data in table (switch to "Data Table" tab)
- Click **"Regression"** - Generate regression chart
- Click **"Histogram"** - Generate histogram
- Type commands in the input area and click **"Send to Python"**

---

## Step 7: Run Web Client (Alternative)

### Step 7.1: Start REST API Server
Make sure `api_server.py` is running (from Step 4, Option B)

### Step 7.2: Open Web Client
1. Open your web browser (Chrome, Firefox, Edge, etc.)
2. Open the file: `web_client.html`
   - **Method 1**: Double-click `web_client.html` in file explorer
   - **Method 2**: Right-click → Open with → Browser
   - **Method 3**: Drag and drop `web_client.html` into browser

### Step 7.3: Connect to API
1. In the web client, verify API URL: `http://127.0.0.1:5000`
2. Click **"Test Connection"**
3. Should show: **"✅ Connected successfully!"**

### Step 7.4: Load Dataset
1. Enter file path in "File Path" field: `data.txt`
2. Click **"Load Dataset"**
3. Should show: **"✅ Loaded: X rows, Y columns"**

### Step 7.5: Test Features
- Click quick action buttons (Summary, Shape, etc.)
- Click visualization buttons (Regression, Histogram, etc.)
- Type commands and click **"Execute"**
- View data in the **"Data Table"** section

---

## Step 8: Test Advanced Features

### Machine Learning
```
# In Java client or via command:
ml_train_regression 2 linear
ml_predict regression_linear_0 2104 3
ml_evaluate regression_linear_0
ml_list
```

### Real-time Analytics
```
# Start streaming
stream_start 2.0

# Set alert
alert_set 2 > 500000

# Get latest data
stream_get 10

# List alerts
alert_list

# Stop streaming
stream_stop
```

### Data Preprocessing
```
# Filter data
filter 0 > 2000

# Sort data
sort 1 desc

# Group data
group 2 mean
```

### Data Cleaning
```
# Remove duplicates
remove_duplicates

# Fill missing values
fillna mean

# Comprehensive cleaning
clean_data
```

### Chart Customization
```
# Customize chart
chart_title My Custom Chart
chart_color blue
chart_size 12,8
chart

# Reset
chart_reset
```

---

## Step 9: Test Database Features

```
# Save dataset to database
db_save my_dataset

# List all datasets
db_list

# Load dataset from database
db_load my_dataset

# Execute SQL query
sql SELECT * FROM datasets

# Delete dataset
db_delete my_dataset
```

---

## Troubleshooting

### Problem: "Module not found" error
**Solution**: Install dependencies
```powershell
pip install -r requirements.txt
```

### Problem: "Port already in use"
**Solution**: 
- Close other instances of the server
- Or change port in `config.json`

### Problem: Java client won't connect
**Solution**:
- Make sure server is running
- Check firewall settings
- Verify host and port match server settings

### Problem: Web client shows connection error
**Solution**:
- Make sure `api_server.py` is running
- Check browser console for errors (F12)
- Verify API URL is correct

### Problem: "File not found" error
**Solution**:
- Make sure data file is in the `base paper implementation` folder
- Use full path: `D:\SEMESTER 7\CNET\PROJECT\base paper implementation\data.txt`

### Problem: Visualizations not showing
**Solution**:
- Make sure dataset is loaded first
- Check that dataset has numeric columns
- Look at server terminal for error messages

### Problem: ML training fails
**Solution**:
- Ensure dataset has enough rows (at least 10)
- Check that target column is numeric
- Verify feature columns exist

---

## Quick Start Summary

**Minimum steps to get started:**

1. Install dependencies: `pip install -r requirements.txt`
2. Start server: `python server.py` (keep running)
3. Compile Java: `javac JavaSwingClient.java`
4. Run Java: `java JavaSwingClient`
5. Load data: Click "Load File..." → Select `data.txt`
6. Test: Click "Summary" or "Regression" button

**That's it!** You should now see results.

---

## Running Multiple Clients

You can run multiple clients simultaneously:

1. **Socket Server** (Terminal 1): `python server.py`
2. **REST API Server** (Terminal 2): `python api_server.py`
3. **Java Client** (Terminal 3): `java JavaSwingClient`
4. **Web Client** (Browser): Open `web_client.html`

All can run at the same time and connect to their respective servers!

---

## Next Steps

Once everything is running:
1. Explore all visualization types
2. Try data preprocessing commands
3. Test machine learning features
4. Set up real-time alerts
5. Export charts and data
6. Save datasets to database

---

## Command Reference

### Basic Commands
- `data.txt` - Load dataset
- `summary` - Statistical summary
- `shape` - Dataset dimensions
- `head` - First 10 rows
- `head 5` - First 5 rows

### Visualizations
- `chart` - Regression chart
- `violin` - Violin plot
- `pair` - Pairplot
- `histogram` - Histogram
- `boxplot` - Boxplot
- `heatmap` - Correlation heatmap
- `barchart` - Bar chart
- `scattermatrix` - Scatter matrix
- `all` - Generate all charts

### Data Operations
- `getdata` - Get data as JSON (for table view)
- `mean`, `median`, `std`, `var` - Statistics
- `corr` - Correlation matrix

### Preprocessing
- `filter 0 > 2000` - Filter rows
- `sort 1 desc` - Sort data
- `group 2 mean` - Group and aggregate

### Cleaning
- `remove_duplicates` - Remove duplicates
- `dropna` - Remove missing values
- `fillna mean` - Fill missing values
- `clean_data` - Comprehensive cleaning

### Transformations
- `normalize` - Min-Max normalization
- `standardize` - Z-score standardization
- `log_transform 0` - Log transformation

### Machine Learning
- `ml_train_regression 2 linear` - Train regression
- `ml_predict model_name 2104 3` - Make prediction
- `ml_evaluate model_name` - Evaluate model
- `ml_importance model_name` - Feature importance
- `ml_list` - List models

### Real-time
- `stream_start 2.0` - Start streaming
- `stream_stop` - Stop streaming
- `stream_get 10` - Get latest data
- `alert_set 2 > 500000` - Set alert
- `alert_list` - List alerts

### Database
- `db_save name` - Save dataset
- `db_load name` - Load dataset
- `db_list` - List datasets
- `db_delete name` - Delete dataset
- `sql SELECT ...` - SQL query

---

**You're all set!** Follow these steps and you'll have the enhanced system running in minutes.





