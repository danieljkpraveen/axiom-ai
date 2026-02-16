import base64
import json
import logging
import os
import re
from io import BytesIO

from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST
from .forms import SignupForm
from .models import ChatAttachment, ChatMessage, ChatSession
from .services import SYSTEM_PROMPT, call_moonshot_with_retry, call_moonshot_with_tools

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_EDGE = 1024
MAX_HISTORY = 8
SMALLTALK_SET = {"hi", "hello", "hey", "yo", "sup", "hola"}
MANDATORY_SEARCH_KEYWORDS = {
    "latest",
    "current",
    "today",
    "now",
    "recent",
    "released",
    "release",
    "version",
    "changelog",
    "price",
    "pricing",
    "ceo",
    "president",
    "law",
    "regulation",
    "news",
    "update",
}


def _static_response_for_query(normalized: str) -> str | None:
    if not normalized:
        return None
    model_intents = {
        "what model are you",
        "which model are you",
        "what model are you running",
        "what model do you use",
        "who built you",
        "who made you",
        "who created you",
        "who are you",
        "are you openai",
    }
    if normalized in model_intents or normalized.startswith("what model") or normalized.startswith("who built"):
        return (
            "I'm Axiom, an AI research assistant. "
            "I don't disclose underlying model or provider details. "
            "If you have a task, I'm ready to help."
        )
    if "seahorse" in normalized and ("show" in normalized or "image" in normalized or "picture" in normalized):
        return "Here’s a seahorse image:\n\n![Seahorse](/static/chat/seahorse.svg)"
    return None


def _strip_sources_block(text: str) -> str:
    lowered = text.lower()
    marker = lowered.find("sources:")
    if marker == -1:
        return text
    return text[:marker].rstrip()


def _normalize_query(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    return " ".join(cleaned.split())


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _research_policy_prompt() -> str:
    today = timezone.now().date().isoformat()
    knowledge_cutoff = os.getenv("MOONSHOT_KNOWLEDGE_CUTOFF", "unknown")
    return (
        f"Runtime date: {today}. "
        f"Model knowledge cutoff may be {knowledge_cutoff}. "
        "If the user asks about events likely after the cutoff or likely to change over time, search first. "
        "If search cannot retrieve enough reliable evidence, explicitly say what is unknown and avoid assumptions."
    )


def _requires_mandatory_search(query: str) -> bool:
    normalized = _normalize_query(query)
    tokens = set(normalized.split())
    if tokens.intersection(MANDATORY_SEARCH_KEYWORDS):
        return True
    if re.search(r"\b(what|which)\s+.*\b(version|release)\b", normalized):
        return True
    if re.search(r"\b(latest|current|newest)\b", normalized):
        return True
    if re.search(r"\b(202[0-9]|19[0-9]{2})\b", normalized):
        return True
    return False


def _compress_image(uploaded_file):
    from PIL import Image

    if uploaded_file.size > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large (max 4MB).")

    try:
        image = Image.open(uploaded_file)
        image.load()
    except Exception as exc:
        raise ValueError("Invalid image file.") from exc

    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
    buffer = BytesIO()
    image.save(buffer, format="WEBP", quality=80, method=6)
    data = buffer.getvalue()
    return data, image.width, image.height, len(data)


def _get_sidebar_sessions(user):
    return ChatSession.objects.filter(user=user).order_by("-updated_at")[:25]


@login_required
def index(request):
    sessions = _get_sidebar_sessions(request.user)
    start_new = request.GET.get("new") == "1"
    active_session = None
    if not start_new and request.GET.get("resume") == "1":
        active_session = sessions[0] if sessions else None
    messages = active_session.messages.all() if active_session else []
    return render(
        request,
        "chat/index.html",
        {
            "sessions": sessions,
            "active_session": active_session,
            "messages": messages,
            "force_new": active_session is None,
        },
    )


@login_required
def session_view(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    sessions = _get_sidebar_sessions(request.user)
    messages = session.messages.all()
    return render(
        request,
        "chat/index.html",
        {
            "sessions": sessions,
            "active_session": session,
            "messages": messages,
        },
    )


@login_required
@require_GET
def chat_messages_partial(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    messages = session.messages.all()
    payload = []
    for message in messages:
        attachments = [
            {"url": attachment.image.url}
            for attachment in message.attachments.all()
        ]
        status = "complete"
        if message.role == "assistant" and not message.content:
            status = "pending"
        payload.append(
            {
                "id": str(message.id),
                "role": message.role,
                "content": message.content,
                "status": status,
                "attachments": attachments,
            }
        )
    return JsonResponse({"messages": payload})


def signup_view(request):
    if request.method == "POST":
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("chat_home")
    else:
        form = SignupForm()
    return render(request, "auth/signup.html", {"form": form})


@login_required
@require_POST
def chat_send(request):
    if request.content_type and request.content_type.startswith("multipart/"):
        message_text = (request.POST.get("message") or "").strip()
        session_id = request.POST.get("session_id") or None
        upload = request.FILES.get("image")
    else:
        data = json.loads(request.body or "{}")
        message_text = (data.get("message") or "").strip()
        session_id = data.get("session_id") or None
        upload = None

    if not message_text and not upload:
        return JsonResponse({"error": "Message cannot be empty."}, status=400)

    if session_id:
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    else:
        session = ChatSession.objects.create(user=request.user)

    user_message = ChatMessage.objects.create(session=session, role="user", content=message_text)

    if not session.title:
        session.title = message_text[:60]
    session.updated_at = timezone.now()
    session.save(update_fields=["title", "updated_at"])

    image_payload = None
    if upload:
        try:
            data, width, height, byte_size = _compress_image(upload)
            attachment = ChatAttachment.objects.create(
                message=user_message,
                image=ContentFile(data, name="upload.webp"),
                image_width=width,
                image_height=height,
                byte_size=byte_size,
            )
            attachment.save()
            image_payload = {
                "mime": "image/webp",
                "data": base64.b64encode(data).decode("ascii"),
            }
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)

    normalized = _normalize_query(message_text)
    if not upload and (normalized in SMALLTALK_SET or len(normalized) <= 2):
        assistant_text = "Hello! Ask me anything you want to research."
        assistant_message = ChatMessage.objects.create(
            session=session,
            role="assistant",
            content=assistant_text,
        )
        session.updated_at = timezone.now()
        session.save(update_fields=["updated_at"])
        return JsonResponse(
            {
                "session_id": str(session.id),
                "assistant_message": assistant_message.content,
                "sources": [],
            }
        )

    if not upload:
        static_response = _static_response_for_query(normalized)
        if static_response:
            assistant_message = ChatMessage.objects.create(
                session=session,
                role="assistant",
                content=static_response,
            )
            session.updated_at = timezone.now()
            session.save(update_fields=["updated_at"])
            return JsonResponse(
                {
                    "session_id": str(session.id),
                    "assistant_message": assistant_message.content,
                    "sources": [],
                }
            )

    assistant_message = ChatMessage.objects.create(
        session=session,
        role="assistant",
        content="",
    )

    image_description = ""
    if image_payload:
        try:
            vision_messages = [
                {"role": "system", "content": "You are a vision assistant. Describe the image content concisely."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message_text or "Describe the image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_payload['mime']};base64,{image_payload['data']}"},
                        },
                    ],
                },
            ]
            image_description = call_moonshot_with_retry(vision_messages)[:500]
        except Exception as exc:
            logger.exception("Moonshot image analysis failed")
            image_description = ""

    recent_messages = (
        ChatMessage.objects.filter(session=session)
        .order_by("-created_at")[:MAX_HISTORY]
    )
    history = list(reversed(recent_messages))
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": _research_policy_prompt()},
    ]
    if image_payload:
        content = []
        text_parts = []
        if message_text:
            text_parts.append(message_text)
        if image_description:
            text_parts.append(f"Image summary: {image_description}")
        if text_parts:
            content.append({"type": "text", "text": "\n".join(text_parts)})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_payload['mime']};base64,{image_payload['data']}"},
            }
        )
        prompt_messages.append({"role": "user", "content": content})
    else:
        for msg in history:
            if msg.role == "assistant" and not msg.content:
                continue
            prompt_messages.append({"role": msg.role, "content": msg.content})

    try:
        if image_payload:
            assistant_text = call_moonshot_with_retry(prompt_messages)
        else:
            search_enabled = _env_flag("MOONSHOT_ENABLE_WEB_SEARCH", default=True)
            if search_enabled and _requires_mandatory_search(message_text):
                prompt_messages.insert(
                    2,
                    {
                        "role": "system",
                        "content": (
                            "Mandatory action: call $web_search before answering this query. "
                            "Use retrieved evidence to answer. If evidence is missing/conflicting, say so."
                        ),
                    },
                )
            search_model = os.getenv("MOONSHOT_SEARCH_MODEL", "moonshot-v1-auto") if search_enabled else None
            assistant_text = call_moonshot_with_tools(
                prompt_messages,
                enable_web_search=search_enabled,
                model_override=search_model,
            )
    except Exception as exc:
        logger.exception("Moonshot request failed")
        assistant_text = "The model is slow or unavailable right now. Please try again in a moment."

    if assistant_text:
        assistant_text = _strip_sources_block(assistant_text)
    if not assistant_text or not str(assistant_text).strip():
        assistant_text = "I couldn't complete a grounded answer for that request. Please try rephrasing."

    # Sources display temporarily disabled

    assistant_message.content = assistant_text
    assistant_message.save(update_fields=["content"])
    session.updated_at = timezone.now()
    session.save(update_fields=["updated_at"])

    return JsonResponse(
        {
            "session_id": str(session.id),
            "assistant_message": assistant_message.content,
            "sources": [],
        }
    )
