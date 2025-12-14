# Complete Step-by-Step Guide to Run the Project

## Step 1: Open Terminal/Command Prompt

1. Press `Windows Key + R`
2. Type `powershell` and press Enter
3. Or search for "PowerShell" in Start menu

---

## Step 2: Navigate to Project Folder

In PowerShell, type:
```powershell
cd "D:\SEMESTER 7\CNET\PROJECT\base paper implementation"
```

**Verify you're in the right folder:**
```powershell
pwd
```
Should show: `D:\SEMESTER 7\CNET\PROJECT\base paper implementation`

**List files to confirm:**
```powershell
ls
```
You should see: `server.py`, `JavaSwingClient.java`, `requirements.txt`, etc.

---

## Step 3: Install Python Dependencies

**Check Python is installed:**
```powershell
python --version
```
Should show: `Python 3.x.x`

**Install required packages:**
```powershell
pip install -r requirements.txt
```

**Wait for installation to complete.** You should see:
```
Successfully installed numpy-... pandas-... matplotlib-... etc.
```

**If you get errors**, try:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 4: Verify Installation

Test that everything is installed:
```powershell
python -c "import pandas, matplotlib, seaborn, sklearn, flask; print('All dependencies OK')"
```

Should output: `All dependencies OK`

---

## Step 5: Start the Python Server

**In the same terminal, run:**
```powershell
python server.py
```

**You should see:**
```
Server listening on 127.0.0.1:1234 ...
Server supports concurrent client connections.
```

**IMPORTANT:** Keep this terminal window open! The server must keep running.

**If you see errors:**
- Make sure you're in the correct folder
- Check that all dependencies are installed
- Look at the error message for clues

---

## Step 6: Open a NEW Terminal Window

**Don't close the server terminal!** Open a **second** PowerShell window:

1. Press `Windows Key + R`
2. Type `powershell` and press Enter
3. Or right-click Start menu → Windows PowerShell

---

## Step 7: Navigate to Project Folder (Again)

In the **new** terminal:
```powershell
cd "D:\SEMESTER 7\CNET\PROJECT\base paper implementation"
```

---

## Step 8: Compile Java Client

**Check Java is installed:**
```powershell
javac -version
```
Should show: `javac 1.x.x` or `javac 17.x.x` etc.

**Compile the Java client:**
```powershell
javac JavaSwingClient.java
```

**Wait for compilation.** If successful, you'll see no errors (just returns to prompt).

**If you see errors:**
- Make sure Java JDK is installed
- Check that you're in the correct folder
- Look at the error messages

---

## Step 9: Run Java Client

**In the same terminal (where you compiled):**
```powershell
java JavaSwingClient
```

**A window should open** showing the Java client interface with:
- Connection settings at the top
- Quick action buttons
- Command input area
- Output area
- Data table tab
- Visualization area at bottom

---

## Step 10: Connect to Server

**In the Java client window:**

1. **Verify connection settings:**
   - Server: `127.0.0.1`
   - Port: `1234`

2. **Click "Test Connection" button**
   - Status bar should show: **"Connection successful!"** in green
   - If it shows red, check that server is running in the other terminal

---

## Step 11: Load Your Dataset

**Option A: Using File Picker (Recommended)**
1. Click **"Load File..."** button (or File → Load Dataset...)
2. Navigate to your data file (e.g., `data.txt`)
3. Select the file and click **Open**
4. Output area should show: `DataFrame 'df' loaded from 'data.txt' with shape (X, Y)`

**Option B: Using Command**
1. Type the file path in the command area: `data.txt`
2. Click **"Send to Python"**
3. Should show the same success message

**If file not found:**
- Make sure `data.txt` is in the `base paper implementation` folder
- Or use full path: `D:\SEMESTER 7\CNET\PROJECT\base paper implementation\data.txt`

---

## Step 12: Test Basic Features

**Try these quick actions:**

1. **Click "Summary" button**
   - Should show statistical summary in output area

2. **Click "Shape" button**
   - Should show dataset dimensions

3. **Click "View Data" button**
   - Switch to "Data Table" tab
   - Should see your data in a table

4. **Click "Regression" button**
   - Should generate and display a regression chart in the bottom panel

5. **Click "Histogram" button**
   - Should generate and display a histogram

---

## Step 13: Try More Features

### Visualizations
- Click any visualization button: **Violin**, **Pairplot**, **Boxplot**, **Heatmap**, **Bar Chart**, **Scatter Matrix**
- Click **"All Charts"** to generate all visualizations

### Data Operations
- Type commands: `head`, `tail`, `mean`, `std`, `corr`
- Try: `head 5` to see first 5 rows

### Data Preprocessing
- Try: `filter 0 > 2000` (filter rows)
- Try: `sort 1 desc` (sort data)
- Try: `group 2 mean` (group and aggregate)

### Export Features
- Generate a chart, then click **"Save Image..."** to export
- View data table, then click **"Export to CSV..."** to save data

---

## Alternative: Run Web Client

**If you prefer a web interface:**

### Step 1: Start REST API Server
Open a **third** terminal window:
```powershell
cd "D:\SEMESTER 7\CNET\PROJECT\base paper implementation"
python api_server.py
```

Should see:
```
Starting REST API server on http://127.0.0.1:5000
* Running on http://127.0.0.1:5000
```

### Step 2: Open Web Client
1. Navigate to the project folder in File Explorer
2. Double-click `web_client.html`
3. It should open in your default browser

### Step 3: Connect
1. In the web client, verify API URL: `http://127.0.0.1:5000`
2. Click **"Test Connection"**
3. Should show: **"✅ Connected successfully!"**

### Step 4: Use Web Client
- Enter file path: `data.txt`
- Click **"Load Dataset"**
- Click any button to test features
- View visualizations in the browser

---

## Troubleshooting

### Problem: "python: command not found"
**Solution**: Python not in PATH. Use full path or reinstall Python with "Add to PATH" option.

### Problem: "javac: command not found"
**Solution**: Java JDK not installed or not in PATH. Install Java JDK.

### Problem: "Module not found" when running server
**Solution**: Dependencies not installed. Run: `pip install -r requirements.txt`

### Problem: "Port already in use"
**Solution**: 
- Another server is running. Close it first.
- Or change port in `config.json`

### Problem: Java client won't connect
**Solution**:
- Make sure server is running (check first terminal)
- Check firewall settings
- Verify host/port match: `127.0.0.1:1234`

### Problem: "File not found" when loading data
**Solution**:
- Make sure data file is in `base paper implementation` folder
- Use full path if needed
- Check file name spelling

### Problem: Visualizations not showing
**Solution**:
- Make sure dataset is loaded first
- Check dataset has numeric columns
- Look at server terminal for error messages

---

## Quick Reference

### Terminal 1 (Server):
```powershell
cd "D:\SEMESTER 7\CNET\PROJECT\base paper implementation"
python server.py
```
**Keep this running!**

### Terminal 2 (Java Client):
```powershell
cd "D:\SEMESTER 7\CNET\PROJECT\base paper implementation"
javac JavaSwingClient.java
java JavaSwingClient
```

### Terminal 3 (Optional - REST API):
```powershell
cd "D:\SEMESTER 7\CNET\PROJECT\base paper implementation"
python api_server.py
```

---

## Summary Checklist

- [ ] Opened terminal
- [ ] Navigated to project folder
- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Started server (`python server.py`)
- [ ] Opened new terminal
- [ ] Compiled Java client (`javac JavaSwingClient.java`)
- [ ] Ran Java client (`java JavaSwingClient`)
- [ ] Tested connection
- [ ] Loaded dataset
- [ ] Tested features

---

## You're Done! 🎉

Once you see the Java client window and can load data and generate visualizations, everything is working correctly!

**Next Steps:**
- Explore all visualization types
- Try data preprocessing commands
- Test machine learning features
- Export charts and data
- Try the web client

**For help with specific features, see:**
- `RUN_GUIDE.md` - Detailed usage guide
- `README.md` - Complete documentation
- `CHANGES.md` - All features implemented




