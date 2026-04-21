# database.py
import sqlite3
import pandas as pd

def get_connection():
    # check_same_thread=False is needed for Streamlit to share the connection safely
    return sqlite3.connect('energy_final_v2.db', check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS appliance_list (name TEXT PRIMARY KEY, watts REAL, saved_hours REAL DEFAULT 0)')
    c.execute('CREATE TABLE IF NOT EXISTS energy_history (year INTEGER, month TEXT, actual_kwh REAL, context TEXT)')
    conn.commit()
    return conn

def fetch_appliances():
    conn = get_connection()
    return pd.read_sql("SELECT * FROM appliance_list", conn)

def fetch_history():
    conn = get_connection()
    return pd.read_sql("SELECT * FROM energy_history", conn)