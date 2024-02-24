from django.test import TestCase

# Create your tests here.
# Create your tests here.
import smtplib

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('rohitsav147@gmail.com', 'ijepopljmlsgvais')
server.sendmail('rohitsav147@gmail.com', 'chintamani.virulkar@gmail.com', 'sub', 'body body body')
print('mail sent')