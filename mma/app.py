import streamlit as st
import pandas as pd
from database import user_db, email_db, logs_db
from auth.gmail_auth import send_email_gmail
from services.bulk_email import send_bulk_emails
from utils.csv_import import import_contacts_csv
import base64
from datetime import datetime
from scheduler import get_scheduled_jobs, cancel_scheduled_email
from genai_email import gen_email
import matplotlib.pyplot as plt
from database.logs_db import get_email_statistics

user_db.create_users_table()
email_db.create_email_tables()
logs_db.create_logs_table()


def get_classification():
    import pandas as pd
    d=pd.read_csv('../classification_result.csv').to_dict('records')
    return d[0]

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ''


st.title("Disaster Dial - Connecting safety, instantly")
menu = ["Login", "SignUp"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Login":
    if st.session_state.logged_in:
        st.sidebar.write(f"Logged in as: {st.session_state.username}")
        selected_action = st.selectbox("Choose Action", ["Send Emails", "Cancel Scheduled Emails", "Import Contacts", "Manage Contacts", "Manage Templates", "View Logs", "Logout"])
        
        if selected_action == "Send Emails":
            service = st.radio("Choose Email Service", ["Gmail", "Outlook"])
            contacts = email_db.view_contacts()
            contact_names = [contact[1] for contact in contacts]
            selected_contact_names = st.multiselect("Select Contacts", contact_names)
            
            selected_contacts = [contact[2] for contact in contacts if contact[1] in selected_contact_names]

            templates = email_db.view_templates()
            template_names = [template[1] for template in templates]
            selected_template = st.selectbox("Choose Template", template_names)
            
            subject = ""
            body = ""
            for template in templates:
                if template[1] == selected_template:
                    subject = template[2]
                    body = template[3]
                    break
            
            schedule_option = st.checkbox("Schedule Email")
            run_date = None
            if schedule_option:
                schedule_date = st.date_input("Choose a date")
                schedule_time = st.time_input("Choose a time")
                run_date = datetime.combine(schedule_date, schedule_time)

            if st.button("Send Emails"):
                if selected_contacts:
                    response = send_bulk_emails(service, selected_contacts, subject, body, run_date)
                    st.success(response)
                else:
                    st.warning("No contacts selected.")
        
        if selected_action == "Cancel Scheduled Emails":
            st.subheader("Scheduled Emails")

            scheduled_jobs = get_scheduled_jobs()
            
            if scheduled_jobs:
                jobs_df = pd.DataFrame(scheduled_jobs, columns=["Job ID", "Service", "Contacts", "Subject", "Body", "Scheduled Date"])
                st.table(jobs_df)

                job_id_to_cancel = st.selectbox("Select Job ID to Cancel", jobs_df["Job ID"].tolist())
                
                if st.button("Cancel Scheduled Email"):
                    cancel_scheduled_email(job_id_to_cancel)
                    st.success(f"Scheduled email with Job ID {job_id_to_cancel} has been canceled.")
                    st.rerun()

            else:
                st.info("No scheduled emails available.")
        
        elif selected_action == "Manage Contacts":
            st.subheader("Add a New Contact")
            name = st.text_input("Name")
            email = st.text_input("Email")
            if st.button("Add Contact"):
                if name and email:
                    try:
                        email_db.add_contact(name, email)
                        st.success(f"Added contact {name}")
                    except sqlite3.IntegrityError:
                        st.error("Email already exists.")
                else:
                    st.error("Please provide both name and email.")

            contacts = email_db.view_contacts()
            if contacts:
                st.write("Existing Contacts:")
                contact_df = pd.DataFrame(contacts, columns=["ID", "Name", "Email"])
                st.table(contact_df)

                selected_contact_id = st.selectbox("Select Contact ID to Update/Delete", contact_df["ID"].tolist())

                contact_to_edit = contact_df[contact_df["ID"] == selected_contact_id]
                if not contact_to_edit.empty:
                    new_name = st.text_input("Update Contact Name", value=contact_to_edit.iloc[0]["Name"])
                    new_email = st.text_input("Update Contact Email", value=contact_to_edit.iloc[0]["Email"])

                    if st.button("Update Contact"):
                        email_db.update_contact(selected_contact_id, new_name, new_email)
                        st.success(f"Contact ID {selected_contact_id} updated successfully.")
                        st.rerun()

                    if st.button("Delete Contact"):
                        email_db.delete_contact(selected_contact_id)
                        st.success(f"Contact ID {selected_contact_id} deleted successfully.")
                        st.rerun()

            else:
                st.info("No contacts available.")
            
            

        if selected_action == "Manage Templates":
            if st.button("Generate using AI"):
                highlighted_categories = [category.replace('_', ' ').title() for category, classification in get_classification().items() if classification == 1]
                if highlighted_categories:
                    email_response = gen_email(highlighted_categories[1:])
                    st.write(email_response)
                else:
                    st.info("No critical issues were highlighted.")
                    
            st.subheader("Add a New Template")
            template_name = st.text_input("Template Name")
            template_subject = st.text_input("Template Subject")
            template_body = st.text_area("Template Body")

            if st.button("Add Template"):
                if template_name and template_subject and template_body:
                    email_db.add_template(template_name, template_subject, template_body)
                    st.success(f"Template '{template_name}' added.")
                else:
                    st.error("Please provide all fields for the template.")

            templates = email_db.view_templates()

            if templates:
                st.write("Existing Templates:")
                template_df = pd.DataFrame(templates, columns=["ID", "Name", "Subject", "Body"])
                st.table(template_df)

                selected_template_id = st.selectbox("Select Template ID to Update/Delete", template_df["ID"].tolist())

                template_to_edit = template_df[template_df["ID"] == selected_template_id]
                if not template_to_edit.empty:
                    new_template_name = st.text_input("Update Template Name", value=template_to_edit.iloc[0]["Name"])
                    new_subject = st.text_input("Update Template Subject", value=template_to_edit.iloc[0]["Subject"])
                    new_body = st.text_area("Update Template Body", value=template_to_edit.iloc[0]["Body"])

                    if st.button("Update Template"):
                        email_db.update_template(selected_template_id, new_template_name, new_subject, new_body)
                        st.success(f"Template ID {selected_template_id} updated successfully.")
                        st.rerun()

                    if st.button("Delete Template"):
                        email_db.delete_template(selected_template_id)
                        st.success(f"Template ID {selected_template_id} deleted successfully.")
                        st.rerun()

            else:
                st.info("No templates available.")

        elif selected_action == "View Logs":
            if 'prev_total_sent' not in st.session_state:
                st.session_state.prev_total_sent = 0
            if 'prev_total_delivered' not in st.session_state:
                st.session_state.prev_total_delivered = 0
            if 'prev_delivery_rate' not in st.session_state:
                st.session_state.prev_delivery_rate = 0
            if 'prev_today_sent' not in st.session_state:
                st.session_state.prev_today_sent = 0
            if 'prev_today_delivered' not in st.session_state:
                st.session_state.prev_today_delivered = 0
            if 'prev_today_delivery_rate' not in st.session_state:
                st.session_state.prev_today_delivery_rate = 0

            def get_color_and_arrow(current, previous):
                if current > previous:
                    return "green", "↑"
                elif current < previous:
                    return "red", "↓"
                else:
                    return "white", ""

            st.title("Email Analytics Dashboard")

            st.subheader("Email Logs")
            logs = logs_db.view_email_logs()
            if logs:
                logs_df = pd.DataFrame(logs, columns=["S.No.", "Receiver Email", "Status", "Date/Time"])
                st.table(logs_df)
            else:
                st.info("No logs available.")

            total_sent, total_delivered, total_failed = get_email_statistics()

            delivery_rate = (total_delivered / total_sent * 100) if total_sent > 0 else 0

            if total_delivered==None:
                today_delivered=0
            if total_failed==None:
                today_failed=0

            sent_color, sent_arrow = get_color_and_arrow(total_sent, st.session_state.prev_total_sent)
            delivered_color, delivered_arrow = get_color_and_arrow(total_delivered, st.session_state.prev_total_delivered)
            rate_color, rate_arrow = get_color_and_arrow(delivery_rate, st.session_state.prev_delivery_rate)

            st.subheader("Email Statistics")
            st.markdown(f"<i>Total Emails Sent: </i><span style='color:{sent_color};'>{total_sent} {sent_arrow}</span>", unsafe_allow_html=True)
            st.markdown(f"<i>Total Emails Delivered: </i><span style='color:{delivered_color};'>{total_delivered} {delivered_arrow}</span>", unsafe_allow_html=True)
            st.markdown(f"<i>Total Emails Failed: </i><span>{total_failed}</span>", unsafe_allow_html=True)
            st.markdown(f"<i>Delivery Rate: </i><span style='color:{rate_color};'>{delivery_rate:.2f}% {rate_arrow}</span>", unsafe_allow_html=True)

            st.session_state.prev_total_sent = total_sent
            st.session_state.prev_total_delivered = total_delivered
            st.session_state.prev_delivery_rate = delivery_rate

            data = {
                'Status': ['Sent', 'Delivered', 'Failed'],
                'Count': [total_sent, total_delivered, total_failed]
            }
            df = pd.DataFrame(data)

            st.subheader("Email Delivery Status Distribution")
            fig, ax = plt.subplots(facecolor='none')
            ax.pie(df['Count'], labels=df['Status'], autopct='%1.1f%%', startangle=90, colors=['blue', 'green', 'red'], textprops={'color': 'white'})
            ax.axis('equal')
            st.pyplot(fig)

            st.subheader("Email Delivery Status Bar Chart")
            fig, ax = plt.subplots(facecolor='none')
            ax.set_facecolor('none')
            bars = ax.bar(df['Status'], df['Count'], color=['blue', 'green', 'red'])
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', color='white')
            ax.tick_params(axis='x', colors='white')
            ax.set_yticks([])
            ax.set_ylabel('Count', color='white')
            ax.set_title('Email Delivery Status', color='white')
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.pyplot(fig)

            today_sent, today_delivered, today_failed = get_email_statistics(for_today=True)
            today_delivery_rate = (today_delivered / today_sent * 100) if today_sent > 0 else 0
            if today_delivered==None:
                today_delivered=0
            if today_failed==None:
                today_failed=0
            
            s_color, s_arrow = get_color_and_arrow(today_sent, st.session_state.prev_today_sent)
            d_color, d_arrow = get_color_and_arrow(today_delivered, st.session_state.prev_today_delivered)
            r_color, r_arrow = get_color_and_arrow(today_delivery_rate, st.session_state.prev_today_delivery_rate)

            st.subheader("Today's Email Statistics")
            st.markdown(f"<i>Total Emails Sent: </i><span style='color:{s_color};'>{today_sent} {s_arrow}</span>", unsafe_allow_html=True)
            st.markdown(f"<i>Total Emails Delivered: </i><span style='color:{d_color};'>{today_delivered} {d_arrow}</span>", unsafe_allow_html=True)
            st.markdown(f"<i>Total Emails Failed: </i><span>{today_failed}</span>", unsafe_allow_html=True)
            st.markdown(f"<i>Delivery Rate: </i><span style='color:{r_color};'>{today_delivery_rate:.2f}% {r_arrow}</span>", unsafe_allow_html=True)

            today_data = {
                'Status': ['Sent', 'Delivered', 'Failed'],
                'Count': [today_sent, today_delivered, today_failed]
            }
            today_df = pd.DataFrame(today_data)

            if today_sent!=0:
                st.subheader("Today's Email Delivery Status")
                fig, ax = plt.subplots(facecolor='none')
                ax.set_facecolor('none')
                bars = ax.bar(today_df['Status'], today_df['Count'], color=['blue', 'green', 'red'])
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', color='white')  # Display count on top of the bar
                ax.tick_params(axis='x', colors='white')
                ax.set_yticks([])
                ax.set_ylabel('Count', color='white')
                ax.set_title("Today's Email Delivery Status", color='white')
                for spine in ax.spines.values():
                    spine.set_visible(False)
                st.pyplot(fig)

            st.session_state.prev_today_sent = today_sent
            st.session_state.prev_today_delivered = today_delivered
            st.session_state.prev_today_delivery_rate = today_delivery_rate

        elif selected_action == "Import Contacts":
            st.subheader("Import Contacts from CSV")
            file = st.file_uploader("Upload CSV", type=["csv"])
            if file:
                try:
                    import_contacts_csv(file)
                    st.success("Contacts imported successfully!")
                except Exception as e:
                    st.error(f"Error importing contacts: {e}")
        
        elif selected_action == "Logout":
            st.session_state.logged_in = False
            st.session_state.username = ''
            st.success("Logged out successfully!")
            st.rerun()
    
    else:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            user = user_db.login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Invalid credentials")

elif choice == "SignUp":
    st.subheader("Create a New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')
    if st.button("SignUp"):
        if new_password != confirm_password:
            st.error("Passwords do not match.")
        elif not new_user or not new_password:
            st.error("Please provide both username and password.")
        else:
            try:
                user_db.register_user(new_user, new_password)
                st.success("Account created successfully! Please go to the Login page.")
            except sqlite3.IntegrityError:
                st.error("Username already exists.")


main_bg = "background/bg.jpg"
main_bg_ext = "jpg"

side_bg = "background/bg.jpg"
side_bg_ext = "jpg"

header_bg = "background/bg.jpg"
header_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
    section[class="main st-emotion-cache-uf99v8 ea3mdgi5"] {{
          background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
      }}
    header[data-testid="stHeader"] {{
          background: url(data:image/{header_bg_ext};base64,{base64.b64encode(open(header_bg, "rb").read()).decode()});
      }}
    </style>
    """,
    unsafe_allow_html=True
)

    