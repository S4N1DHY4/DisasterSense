import sqlite3
from datetime import datetime,timezone

def create_logs_table():
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS email_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipient TEXT NOT NULL,
            status TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

def log_email_status(recipient, status):
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()
    c.execute('INSERT INTO email_logs (recipient, status) VALUES (?, ?)', (recipient, status))
    conn.commit()
    conn.close()

def view_email_logs():
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()
    c.execute('SELECT * FROM email_logs')
    logs = c.fetchall()
    conn.close()
    return logs

def get_email_statistics(for_today=False):
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()

    if for_today:
        today_date = datetime.now(timezone.utc).date().isoformat()
        c.execute('''
            SELECT 
                COUNT(*) AS total_sent,
                SUM(CASE WHEN status = "Delivered" THEN 1 ELSE 0 END) AS total_delivered,
                SUM(CASE WHEN status = "Failed" THEN 1 ELSE 0 END) AS total_failed
            FROM email_logs
            WHERE DATE(timestamp) = ?
        ''', (today_date,))
    else:
        c.execute('''
            SELECT 
                COUNT(*) AS total_sent,
                SUM(CASE WHEN status = "Delivered" THEN 1 ELSE 0 END) AS total_delivered,
                SUM(CASE WHEN status = "Failed" THEN 1 ELSE 0 END) AS total_failed
            FROM email_logs
        ''')

    stats = c.fetchone()
    conn.close()
    return stats
