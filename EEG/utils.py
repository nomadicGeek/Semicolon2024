from django.core.mail import send_mail
from django.conf import settings

def send_prediction_email():
    user_email = 'chintamani.virulkar@gmail.com'
    subject = 'Your Prediction Result'
    message = f'Hello,\n\nYour prediction result is: '
    email_from = settings.EMAIL_HOST_USER
    recipient_list = [user_email]
    send_mail(subject, message, email_from, recipient_list)