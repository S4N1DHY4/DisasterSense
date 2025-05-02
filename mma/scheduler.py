from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from apscheduler.triggers.date import DateTrigger
import sqlite3

scheduler = BackgroundScheduler()
scheduler.start()

def create_jobs_table():
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS scheduled_jobs (
            job_id TEXT PRIMARY KEY,
            service TEXT,
            contacts TEXT,
            subject TEXT,
            body TEXT,
            run_date TEXT
        )
    ''')
    conn.commit()
    conn.close()

def schedule_email(job_id, func, run_date, service, contacts, subject, body):
    trigger = DateTrigger(run_date=run_date)
    scheduler.add_job(func, trigger, id=job_id)

    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()
    c.execute('INSERT INTO scheduled_jobs (job_id, service, contacts, subject, body, run_date) VALUES (?, ?, ?, ?, ?, ?)', 
              (job_id, service, ','.join(contacts), subject, body, run_date))
    conn.commit()
    conn.close()

def cancel_scheduled_email(job_id):
    scheduler.remove_job(job_id)
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()
    c.execute('DELETE FROM scheduled_jobs WHERE job_id = ?', (job_id,))
    conn.commit()
    conn.close()

def get_scheduled_jobs():
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()
    c.execute('SELECT * FROM scheduled_jobs')
    jobs = c.fetchall()
    conn.close()
    return jobs

def job_listener(event):
    if event.exception:
        print(f"Job {event.job_id} failed")
    else:
        print(f"Job {event.job_id} completed successfully")
        conn = sqlite3.connect('email_system.db')
        c = conn.cursor()
        c.execute('DELETE FROM scheduled_jobs WHERE job_id = ?', (event.job_id,))
        conn.commit()
        conn.close()

scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

create_jobs_table()