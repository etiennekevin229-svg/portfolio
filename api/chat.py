"""
API de Sebastian public — POST /api/chat

Reçoit : {"messages": [{"role": "user" | "assistant", "text": "..."}]}
Renvoie : {"reply": "..."}

Protections :
- seules les pages du portfolio peuvent l'appeler depuis un navigateur (Origin) ;
- limite de requêtes par visiteur (best effort, en mémoire) ;
- taille des messages et de l'historique bornée ;
- erreurs détaillées uniquement dans les logs Vercel, jamais chez le visiteur.
"""
import json
import os
import sys
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib import llm_router                      # noqa: E402
from lib.sebastian_public import build_prompt   # noqa: E402

MAX_MESSAGE_CHARS = 1000      # longueur max d'une question
MAX_HISTORY = 10              # messages précédents envoyés au modèle
MAX_USER_MESSAGES = 20        # questions max par conversation
RATE_LIMIT = (8, 60)          # 8 questions par minute et par visiteur
MAX_BODY_BYTES = 64_000

# Domaines autorisés en plus du domaine du site lui-même (séparés par des virgules),
# par exemple : https://kevin-etienne.vercel.app,https://etiennekevin229-svg.github.io
EXTRA_ORIGINS = {o.strip().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()}

UNAVAILABLE = ("Je suis momentanément indisponible. Réessaie dans quelques minutes, "
               "ou contacte Kevin directement via la section Contact.")

_hits: dict[str, deque] = defaultdict(deque)


def _rate_limited(ip: str) -> bool:
    limit, window = RATE_LIMIT
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= limit:
        return True
    q.append(now)
    return False


def _clean_history(messages) -> tuple[list[dict], str] | None:
    """Valide et nettoie l'historique reçu du navigateur. None si invalide."""
    if not isinstance(messages, list) or not messages:
        return None
    clean = []
    for m in messages[-(MAX_HISTORY + 1):]:
        if not isinstance(m, dict):
            return None
        role, text = m.get("role"), m.get("text")
        if role not in ("user", "assistant") or not isinstance(text, str):
            return None
        text = text.strip()[:MAX_MESSAGE_CHARS * 4 if role == "assistant" else MAX_MESSAGE_CHARS]
        if text:
            clean.append({"role": role, "text": text})
    if not clean or clean[-1]["role"] != "user":
        return None
    latest = clean.pop()["text"]
    while clean and clean[0]["role"] != "user":
        clean.pop(0)
    return clean, latest


class handler(BaseHTTPRequestHandler):

    def _send(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _origin_allowed(self) -> bool:
        origin = (self.headers.get("Origin") or "").rstrip("/")
        if not origin:
            return True  # appel sans navigateur : laissé passer, la limite de débit s'applique
        host = (self.headers.get("x-forwarded-host") or self.headers.get("Host") or "").lower()
        return urlparse(origin).netloc.lower() == host or origin in EXTRA_ORIGINS

    def do_GET(self):
        providers = llm_router.available_providers()
        if "diag=1" not in (self.path or ""):
            return self._send(200, {"status": "ok", "assistant": "Sebastian", "providers": len(providers)})

        # Diagnostic : teste chaque modèle avec une question minuscule et renvoie
        # « ok » ou le début du message d'erreur (jamais les clés). Limité par IP.
        ip = (self.headers.get("x-forwarded-for") or self.client_address[0]).split(",")[0].strip()
        if _rate_limited(ip):
            return self._send(429, {"error": "rate"})
        results = {}
        for name, run in llm_router._CHAIN:
            try:
                text = run("Réponds seulement : OK", [], "Test", [], lambda n, a: "")
                results[name] = "ok : " + text.strip()[:40]
            except Exception as e:
                results[name] = "erreur : " + (str(e).splitlines()[0][:220] if str(e) else type(e).__name__)
        self._send(200, {"providers": providers, "diagnostic": results})

    def do_POST(self):
        if not self._origin_allowed():
            return self._send(403, {"error": "origin"})

        ip = (self.headers.get("x-forwarded-for") or self.client_address[0]).split(",")[0].strip()
        if _rate_limited(ip):
            return self._send(429, {"error": "rate",
                                    "reply": "Tu poses beaucoup de questions d'un coup 🙂 "
                                             "Attends une minute et réessaie."})

        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("taille")
            data = json.loads(self.rfile.read(length))
            parsed = _clean_history(data.get("messages"))
            if parsed is None:
                raise ValueError("format")
        except Exception:
            return self._send(400, {"error": "bad_request"})

        history, latest = parsed
        if sum(1 for m in history if m["role"] == "user") >= MAX_USER_MESSAGES:
            return self._send(200, {"reply": "Tu as atteint la limite de questions pour cette "
                                             "conversation. Pour aller plus loin, contacte Kevin "
                                             "via la section Contact 🙂", "limit": True})
        try:
            reply, provider = llm_router.generate_response(
                system=build_prompt(), history=history, latest_message=latest,
                tools=[], execute_tool=lambda name, args: "Outil indisponible.",
            )
            print(f"[sebastian] réponse via {provider}")
            return self._send(200, {"reply": reply})
        except Exception as e:  # détail uniquement dans les logs Vercel
            print(f"[sebastian] échec : {e}")
            return self._send(503, {"error": "unavailable", "reply": UNAVAILABLE})
