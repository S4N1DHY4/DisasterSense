from auth.gmail_auth import send_email_gmail
from auth.outlook_win32 import send_email_outlook_win32
from database.logs_db import log_email_status
from datetime import datetime
from scheduler import schedule_email

def send_bulk_emails(service, contacts, subject, body, run_date=None):
    def send_emails():
        for email in contacts:
            try:
                if service == 'Gmail':
                    send_email_gmail(email, subject, body)
                elif service == 'Outlook':
                    send_email_outlook_win32(email, subject, body)
                
                log_email_status(email, "Delivered")
            except:
                log_email_status(email, "Failed")



    if run_date:
        job_id = f"{service}_{datetime.now().timestamp()}"
        schedule_email(job_id, send_emails, run_date, service, contacts, subject, body)
        return f"Email scheduled for {run_date}"
    else:
        send_emails()
        return "Email sent immediately"


