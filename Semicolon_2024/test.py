import os
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Semicolon2024.settings')

# Configure settings
settings.configure(
    EMAIL_BACKEND='django.core.mail.backends.console.EmailBackend')

# Now you can access settings
print(settings.EMAIL_BACKEND)