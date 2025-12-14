"""
Database integration module for SQLite support
"""

import sqlite3
import pandas as pd
from pathlib import Path
import logging

DB_FILE = "visualization_data.db"

def init_database():
    """Initialize database and create tables if they don't exist"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create datasets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            rows INTEGER,
            columns INTEGER
        )
    ''')
    
    # Create dataset_data table (stores actual data)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dataset_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            row_index INTEGER,
            column_index INTEGER,
            value TEXT,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        )
    ''')
    
    # Create operations_history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS operations_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            operation TEXT,
            command TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    logging.info("Database initialized")

def save_dataset_to_db(name: str, df: pd.DataFrame, description: str = "") -> int:
    """Save a pandas DataFrame to the database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Insert dataset metadata
        cursor.execute('''
            INSERT OR REPLACE INTO datasets (name, description, rows, columns)
            VALUES (?, ?, ?, ?)
        ''', (name, description, len(df), len(df.columns)))
        
        dataset_id = cursor.lastrowid
        
        # Clear existing data for this dataset
        cursor.execute('DELETE FROM dataset_data WHERE dataset_id = ?', (dataset_id,))
        
        # Insert data
        for row_idx, row in df.iterrows():
            for col_idx, value in enumerate(row):
                cursor.execute('''
                    INSERT INTO dataset_data (dataset_id, row_index, column_index, value)
                    VALUES (?, ?, ?, ?)
                ''', (dataset_id, int(row_idx), col_idx, str(value)))
        
        conn.commit()
        logging.info(f"Dataset '{name}' saved to database (ID: {dataset_id})")
        return dataset_id
    except Exception as e:
        conn.rollback()
        logging.error(f"Error saving dataset: {e}")
        raise
    finally:
        conn.close()

def load_dataset_from_db(name: str) -> pd.DataFrame:
    """Load a dataset from the database"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        # Get dataset metadata
        cursor = conn.cursor()
        cursor.execute('SELECT id, rows, columns FROM datasets WHERE name = ?', (name,))
        result = cursor.fetchone()
        
        if not result:
            raise ValueError(f"Dataset '{name}' not found in database")
        
        dataset_id, num_rows, num_cols = result
        
        # Load data
        query = '''
            SELECT row_index, column_index, value
            FROM dataset_data
            WHERE dataset_id = ?
            ORDER BY row_index, column_index
        '''
        data = pd.read_sql_query(query, conn, params=(dataset_id,))
        
        # Reshape data into DataFrame
        if data.empty:
            return pd.DataFrame()
        
        # Create DataFrame from pivoted data
        pivot_data = data.pivot(index='row_index', columns='column_index', values='value')
        df = pd.DataFrame(pivot_data.values, columns=[f'col_{i}' for i in range(num_cols)])
        
        # Try to convert to numeric where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        
        logging.info(f"Dataset '{name}' loaded from database ({len(df)} rows, {len(df.columns)} columns)")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise
    finally:
        conn.close()

def list_datasets() -> list:
    """List all datasets in the database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT name, description, rows, columns, created_at
        FROM datasets
        ORDER BY created_at DESC
    ''')
    
    datasets = []
    for row in cursor.fetchall():
        datasets.append({
            'name': row[0],
            'description': row[1] or '',
            'rows': row[2],
            'columns': row[3],
            'created_at': row[4]
        })
    
    conn.close()
    return datasets

def delete_dataset(name: str) -> bool:
    """Delete a dataset from the database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Get dataset ID
        cursor.execute('SELECT id FROM datasets WHERE name = ?', (name,))
        result = cursor.fetchone()
        
        if not result:
            return False
        
        dataset_id = result[0]
        
        # Delete data
        cursor.execute('DELETE FROM dataset_data WHERE dataset_id = ?', (dataset_id,))
        cursor.execute('DELETE FROM operations_history WHERE dataset_id = ?', (dataset_id,))
        cursor.execute('DELETE FROM datasets WHERE id = ?', (dataset_id,))
        
        conn.commit()
        logging.info(f"Dataset '{name}' deleted from database")
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Error deleting dataset: {e}")
        return False
    finally:
        conn.close()

def execute_sql_query(query: str) -> pd.DataFrame:
    """Execute a SQL query and return results as DataFrame"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        # Security: Only allow SELECT queries
        query_upper = query.strip().upper()
        if not query_upper.startswith('SELECT'):
            raise ValueError("Only SELECT queries are allowed for security")
        
        df = pd.read_sql_query(query, conn)
        logging.info(f"SQL query executed: {len(df)} rows returned")
        return df
    except Exception as e:
        logging.error(f"Error executing SQL query: {e}")
        raise
    finally:
        conn.close()

def log_operation(dataset_name: str, operation: str, command: str):
    """Log an operation to the history"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Get dataset ID
        cursor.execute('SELECT id FROM datasets WHERE name = ?', (dataset_name,))
        result = cursor.fetchone()
        
        if result:
            dataset_id = result[0]
            cursor.execute('''
                INSERT INTO operations_history (dataset_id, operation, command)
                VALUES (?, ?, ?)
            ''', (dataset_id, operation, command))
            conn.commit()
    except Exception as e:
        logging.error(f"Error logging operation: {e}")
    finally:
        conn.close()

# Initialize database on import
init_database()





