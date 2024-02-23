import openai
from django.core.mail import send_mail

openai.api_key = "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"
age = '35'
gender = 'Male'
employeeName = 'Chintamani'
prompt = "draft me personalised email to send as a report of mental wellness of employee with name whos is receieint "+employeeName+" in json format with keys Subject and body with recepient name, email body will have that employee's mental health is not good to work and sugestion to deal with mental illness"
print(prompt)
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {"role": "system", "content": prompt}
    ])

output = response.choices[0]['message']['content']
dict = eval(output)
print(output)
print(dict)

