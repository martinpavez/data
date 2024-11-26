import random
import smtplib
from email.mime.text import MIMEText

# Function to assign Secret Santa pairs
def assign_secret_santa(participants):
    givers = participants[:]
    receivers = participants[:]
    
    # Shuffle until no one is their own Secret Santa
    while any(giver == receiver for giver, receiver in zip(givers, receivers)):
        random.shuffle(receivers)
    
    return dict(zip(givers, receivers))

# Function to send email
def send_email(sender_email, sender_password, recipient_email, message):
    try:
        msg = MIMEText(message)
        msg['Subject'] = "Your Secret Santa Assignment!"
        msg['From'] = sender_email
        msg['To'] = recipient_email

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        print(f"Email sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email to {recipient_email}: {e}")

# Main function
def main():
    # Participants and their emails
    participants = {
        "Ami": "amarantapavez20@gmail.com",
        "Nacho": "ignaciopavez54@gmail.com",
        "Loncho": "apavez@ucn.cl",
        "Mami": "r.vasquez2023@yahoo.com",
        "Martin": "martinpavez.3@gmail.com",
        "Picho": "vicentepavez.v@gmail.com",
        "Tean": "pavez.wow@gmail.com",
        "Richi": "ricardo.vasquez.a@ug.uchile.cl"
    }

    # Your email credentials
    sender_email = "martinpavez.3@gmail.com"
    sender_password = "qzki ucbz obsx ygjn"  # Use app-specific password if needed

    # Assign Secret Santa pairs
    assignments = assign_secret_santa(list(participants.keys()))

    # Notify each participant
    for giver, receiver in assignments.items():
        recipient_email = participants[giver]
        message = f"Hi {giver},\n\nYou are the Secret Santa for {receiver}!\n\nHappy gifting!"
        send_email(sender_email, sender_password, recipient_email, message)

if __name__ == "__main__":
    main()