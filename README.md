# Python Data Visualization Server - Enhanced Implementation

## Overview
This is an enhanced version of the base paper implementation with significant improvements in security, functionality, user experience, and code quality.

## Key Improvements

### 1. Security Enhancements ✅
- **Removed `eval()` vulnerability**: Replaced dangerous `eval()` with a safe command parser using whitelisted operations
- **File path validation**: Prevents directory traversal attacks
- **Input validation**: Command length limits and encoding validation
- **File size limits**: Prevents loading excessively large files

### 2. Error Handling & Logging ✅
- **Comprehensive logging**: All operations logged to `server.log` and console
- **Better error messages**: User-friendly error messages with detailed logging
- **Connection timeouts**: Prevents hanging connections
- **Exception handling**: Graceful error handling throughout

### 3. New Visualization Types ✅
- **Histogram**: Distribution visualization
- **Boxplot**: Statistical distribution analysis
- **Correlation Heatmap**: Relationship visualization between numeric columns
- All original visualizations (regression, violin, pairplot) still supported

### 4. Concurrent Client Support ✅
- **Multi-threading**: Server now handles multiple clients simultaneously
- **Thread-safe operations**: Safe concurrent access to shared resources
- **Connection pooling**: Efficient connection management

### 5. Enhanced Java Client UI ✅
- **File picker**: Easy dataset loading via file dialog
- **Quick action buttons**: One-click access to common operations
- **Connection settings**: Configurable server host and port
- **Connection status**: Visual indicator of connection state
- **Better layout**: Improved organization and user experience
- **Image controls**: Clear image functionality
- **Status bar**: Real-time status updates

### 6. Configuration Management ✅
- **JSON configuration**: Easy server configuration via `config.json`
- **Configurable settings**: Host, port, timeouts, file limits, etc.
- **Default values**: Sensible defaults if config file missing

## Installation

### Prerequisites
- Python 3.7+
- Java 8+ (for Java clients)
- Required Python packages (see `requirements.txt`)

### Setup
1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Configure server settings in `config.json`

3. Start the server:
```bash
python server.py
```

4. Run Java client:
```bash
javac JavaSwingClient.java
java JavaSwingClient
```

## Usage

### Server Commands

#### Data Loading
- Load dataset: `data.txt` or `data.csv`
- The file path must be within the allowed base directory

#### Data Operations
- `summary` - Statistical summary of the dataset
- `shape` - Dataset dimensions
- `columns` - Column names
- `head` or `head <n>` - First n rows (default 10, max 100)
- `tail` or `tail <n>` - Last n rows (default 10, max 100)
- `dtypes` - Data types of columns
- `info` - Dataset information
- `mean`, `median`, `std`, `min`, `max` - Statistical operations
- `isnull` - Missing value counts
- `nunique` - Unique value counts per column

#### Visualizations
- `chart` - Linear regression plot
- `violin` - Violin plot
- `pair` - Pairplot
- `histogram` - Histogram
- `boxplot` - Boxplot
- `heatmap` - Correlation heatmap
- `all` - Generate all visualizations

### Java Client Features

1. **Connection Settings**: Configure server host and port
2. **Test Connection**: Verify server connectivity
3. **Load File**: Use file picker to select dataset
4. **Quick Actions**: Buttons for common operations
5. **Command Input**: Manual command entry
6. **Output Display**: View server responses
7. **Visualization Display**: View generated charts
8. **Status Bar**: Monitor connection and operation status

## Configuration

Edit `config.json` to customize:

```json
{
    "server": {
        "host": "127.0.0.1",      // Server host
        "port": 1234,              // Server port
        "max_connections": 5,     // Max concurrent clients
        "timeout_seconds": 30     // Connection timeout
    },
    "security": {
        "allowed_base_dir": ".",  // Base directory for file access
        "max_file_size_mb": 50,   // Maximum file size in MB
        "max_command_length": 1000 // Maximum command length
    },
    "logging": {
        "level": "INFO",          // Log level (DEBUG, INFO, WARNING, ERROR)
        "log_file": "server.log"  // Log file path
    }
}
```

## Security Features

1. **Safe Command Parser**: Only whitelisted operations allowed
2. **Path Validation**: Prevents directory traversal
3. **File Size Limits**: Prevents resource exhaustion
4. **Input Validation**: Command length and encoding checks
5. **Timeout Protection**: Prevents hanging connections

## Logging

All operations are logged to:
- Console output (INFO level and above)
- `server.log` file (all levels)

Log entries include:
- Timestamps
- Log levels
- Client addresses
- Commands executed
- Errors and exceptions

## File Structure

```
base paper implementation/
├── server.py              # Enhanced Python server
├── JavaSwingClient.java   # Enhanced Java Swing client
├── JavaFXClient.java     # JavaFX client (original)
├── JavaAppletClient.java # Applet client (original)
├── config.json           # Server configuration
├── requirements.txt      # Python dependencies
├── data.txt             # Sample dataset
└── README.md           # This file
```

## Troubleshooting

### Server won't start
- Check if port is already in use
- Verify Python dependencies are installed
- Check `server.log` for errors

### Connection refused
- Ensure server is running
- Verify host and port in client match server config
- Check firewall settings

### File not found
- Ensure file path is within allowed base directory
- Check file permissions
- Verify file exists

### Visualization errors
- Ensure dataset is loaded first
- Check dataset has required numeric columns
- Review server logs for detailed errors

## Future Enhancements

Potential improvements for future versions:
- REST API wrapper
- Web-based client
- Docker containerization
- Database integration
- More visualization types
- Data export functionality
- Advanced statistical operations

## License

This implementation is based on the original paper implementation with enhancements for security, functionality, and usability.





