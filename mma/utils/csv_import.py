import pandas as pd
from database.email_db import add_contact

def import_contacts_csv(file):
    try:
        data = pd.read_csv(file)

        if 'name' in data.columns and 'email' in data.columns:
            for index, row in data.iterrows():
                add_contact(row['name'], row['email'])
            print("Contacts successfully imported.")
        else:
            print("CSV file must contain 'name' and 'email' columns.")

    except Exception as e:
        print(f"Error importing contacts: {e}")
