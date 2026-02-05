import uuid

from django.conf import settings
from django.db import models


class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self) -> str:
        return self.title or f"Chat {self.created_at:%Y-%m-%d %H:%M}"


class ChatMessage(models.Model):
    ROLE_CHOICES = [
        ('system', 'System'),
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self) -> str:
        return f"{self.role}: {self.content[:30]}"


def chat_attachment_path(instance: "ChatAttachment", filename: str) -> str:
    return f"chat/{instance.message.session_id}/{uuid.uuid4().hex}.webp"


class ChatAttachment(models.Model):
    message = models.ForeignKey(ChatMessage, on_delete=models.CASCADE, related_name='attachments')
    image = models.ImageField(upload_to=chat_attachment_path)
    image_width = models.PositiveIntegerField()
    image_height = models.PositiveIntegerField()
    byte_size = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

# Create your models here.
