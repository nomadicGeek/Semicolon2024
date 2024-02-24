import smtplib

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('hackhivesemicolon@gmail.com', 'ufbtwmlswjadklak')
server.sendmail('hackhivesemicolon@gmail.com', 'chintamani.virulkar@gmail.com', 'sub', 'body body body')
print('mail sent')