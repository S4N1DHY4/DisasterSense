import sqlite3

def create_email_tables():
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            subject TEXT NOT NULL,
            body TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

def add_contact(name, email):
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()
    c.execute('INSERT INTO contacts (name, email) VALUES (?, ?)', (name, email))
    conn.commit()
    conn.close()

def view_contacts():
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()
    c.execute('SELECT * FROM contacts')
    contacts = c.fetchall()
    conn.close()
    return contacts

def update_contact(contact_id, new_name, new_email):
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()
    c.execute('UPDATE contacts SET name = ?, email = ? WHERE id = ?', (new_name, new_email, contact_id))
    conn.commit()
    conn.close()

def delete_contact(contact_id):
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()
    c.execute('DELETE FROM contacts WHERE id = ?', (contact_id,))
    conn.commit()
    conn.close()


def add_template(name, subject, body):
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()
    c.execute('INSERT INTO templates (name, subject, body) VALUES (?, ?, ?)', (name, subject, body))
    conn.commit()
    conn.close()

def view_templates():
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()
    c.execute('SELECT * FROM templates')
    templates = c.fetchall()
    conn.close()
    return templates

def update_template(template_id, new_name, new_subject, new_body):
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()
    c.execute('UPDATE templates SET name = ?, subject = ?, body = ? WHERE id = ?', (new_name, new_subject, new_body, template_id))
    conn.commit()
    conn.close()

def delete_template(template_id):
    conn = sqlite3.connect('email_system.db')
    c = conn.cursor()
    c.execute('DELETE FROM templates WHERE id = ?', (template_id,))
    conn.commit()
    conn.close()