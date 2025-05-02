import win32com.client as win32
import pythoncom

def send_email_outlook_win32(recipient_email, subject, body):
    try:
        pythoncom.CoInitialize()
        outlook = win32.Dispatch('outlook.application')
        mail = outlook.CreateItem(0)
        
        mail.To = recipient_email
        mail.Subject = subject
        mail.Body = body
        
        mail.Send()
        print(f"Email successfully sent to {recipient_email}")

    except Exception as e:
        print(f"Error sending email: {e}")
