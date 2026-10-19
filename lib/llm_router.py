"""
Routeur multi-LLM de Sebastian : une chaîne de secours gratuite.

Sebastian essaie les fournisseurs dans cet ordre et passe au suivant
si l'un échoue (quota atteint, panne, clé invalide) :

  1. Gemini principal            (GEMINI_API_KEY)
  2. Gemini de secours           (même clé, autre modèle = autre quota gratuit)
  3. GitHub Models               (GITHUB_TOKEN)      — optionnel
  4. Groq                        (GROQ_API_KEY)      — optionnel
  5. Mistral                     (MISTRAL_API_KEY)   — optionnel

Un fournisseur sans clé est simplement ignoré.
Gemini utilise le SDK officiel « google-genai ». Les autres parlent tous
le format OpenAI, donc un seul morceau de code les gère (librairie « openai »).
"""
import os
import json
from google import genai
from google.genai import types
from openai import OpenAI

MAX_TOOL_ROUNDS = 5


def _secret(name: str) -> str | None:
    """Lit une valeur dans .env (sur ton PC) ou dans les Secrets de Streamlit Cloud."""
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        import streamlit as st
        value = st.secrets.get(name)
        return value.strip() if value else None
    except Exception:
        return None


# ---------- Configuration (modifiable dans .env sans toucher au code) ----------

GEMINI_MODEL = _secret("GEMINI_MODEL") or "gemini-3.6-flash"
GEMINI_BACKUP_MODEL = _secret("GEMINI_BACKUP_MODEL") or "gemini-3.5-flash-lite"

# Fournisseurs au format OpenAI : (nom, variable de la clé, adresse, modèle par défaut,
#                                   nb max de messages d'historique envoyés)
OPENAI_COMPATIBLE = [
    # GitHub Models gratuit : ~150 requêtes/jour, mais 8 000 tokens max par requête
    # → on envoie moins d'historique.
    ("github", "GITHUB_TOKEN", "https://models.github.ai/inference",
     _secret("GITHUB_MODEL") or "openai/gpt-4.1-mini", 6),
    # llama-3.3-70b-versatile a été retiré par Groq le 16/08/2026 → remplaçant officiel
    ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1",
     _secret("GROQ_MODEL") or "openai/gpt-oss-120b", 20),
    ("mistral", "MISTRAL_API_KEY", "https://api.mistral.ai/v1",
     _secret("MISTRAL_MODEL") or "mistral-small-latest", 20),
]

_gemini_key = _secret("GEMINI_API_KEY")
_gemini = genai.Client(api_key=_gemini_key) if _gemini_key else None


def _trim(history: list, limit: int) -> list:
    """Garde les `limit` derniers messages, en commençant par un message utilisateur."""
    recent = history[-limit:] if limit else []
    while recent and recent[0]["role"] != "user":
        recent = recent[1:]
    return recent


# ---------- Gemini ----------

def _gemini_config(system: str, tools: list) -> types.GenerateContentConfig:
    config = types.GenerateContentConfig(system_instruction=system)
    if tools:
        config.tools = [types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters_json_schema=t["input_schema"],
            )
            for t in tools
        ])]
        # On exécute les outils nous-mêmes (boucle ci-dessous)
        config.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
    return config


def _run_gemini(model: str, system: str, history: list, latest_message: str,
                tools: list, execute_tool) -> str:
    if _gemini is None:
        raise RuntimeError("pas de clé GEMINI_API_KEY")

    # L'objet « chat » conserve les réponses complètes du modèle, y compris les
    # « thought signatures » exigées par Gemini 3 lors des appels d'outils.
    chat = _gemini.chats.create(
        model=model,
        config=_gemini_config(system, tools),
        history=[
            types.Content(role="model" if h["role"] == "assistant" else "user",
                          parts=[types.Part(text=h["text"])])
            for h in history
        ],
    )
    response = chat.send_message(latest_message)

    for _ in range(MAX_TOOL_ROUNDS):
        calls = response.function_calls or []
        if not calls:
            break
        response = chat.send_message([
            types.Part.from_function_response(
                name=fc.name,
                response={"result": execute_tool(fc.name, dict(fc.args or {}))},
            )
            for fc in calls
        ])
    else:
        if response.function_calls:
            return "Désolé, je n'ai pas réussi à conclure après plusieurs appels d'outils."

    text = response.text or ""
    if not text.strip():
        raise RuntimeError("réponse vide")
    return text


# ---------- Fournisseurs au format OpenAI (GitHub, Groq, Mistral) ----------

def _run_openai_compatible(client: OpenAI, model: str, system: str, history: list,
                           latest_message: str, tools: list, execute_tool) -> str:
    messages = [{"role": "system", "content": system}]
    messages += [{"role": h["role"], "content": h["text"]} for h in history]
    messages.append({"role": "user", "content": latest_message})

    tool_defs = [
        {"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in tools
    ] if tools else None

    for _ in range(MAX_TOOL_ROUNDS):
        kwargs = {"model": model, "messages": messages}
        if tool_defs:
            kwargs["tools"] = tool_defs
        response = client.chat.completions.create(**kwargs)
        if isinstance(response, str):
            # Le service a répondu autre chose que du JSON (message d'erreur en texte brut)
            raise RuntimeError("réponse inattendue du service : " + response.strip()[:200])
        msg = response.choices[0].message

        if not msg.tool_calls:
            text = msg.content or ""
            if not text.strip():
                raise RuntimeError("réponse vide")
            return text

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result[:4000]})

    return "Désolé, je n'ai pas réussi à conclure après plusieurs appels d'outils."


# ---------- Chaîne de secours ----------

def _build_chain() -> list:
    """Liste des fournisseurs disponibles : (étiquette, fonction d'appel)."""
    chain = []
    if _gemini:
        chain.append(("gemini", lambda s, h, m, t, e: _run_gemini(GEMINI_MODEL, s, h, m, t, e)))
        if GEMINI_BACKUP_MODEL and GEMINI_BACKUP_MODEL != GEMINI_MODEL:
            chain.append(("gemini-lite", lambda s, h, m, t, e:
                          _run_gemini(GEMINI_BACKUP_MODEL, s, h, m, t, e)))
    for name, key_var, base_url, model, max_history in OPENAI_COMPATIBLE:
        key = _secret(key_var)
        if not key:
            continue
        client = OpenAI(api_key=key, base_url=base_url, timeout=60, max_retries=1)
        chain.append((name, lambda s, h, m, t, e, c=client, mo=model, mh=max_history:
                      _run_openai_compatible(c, mo, s, _trim(h, mh), m, t, e)))
    return chain


_CHAIN = _build_chain()


def available_providers() -> list[str]:
    """Noms des fournisseurs configurés, dans l'ordre d'essai."""
    return [name for name, _ in _CHAIN]


def generate_response(system: str, history: list, latest_message: str, tools: list, execute_tool):
    """
    Essaie chaque fournisseur de la chaîne jusqu'à ce que l'un réponde.
    Retourne (texte_de_reponse, fournisseur_utilise).
    """
    if not _CHAIN:
        raise RuntimeError("Aucune clé API configurée dans .env")

    # Si un fournisseur plante après avoir utilisé un outil, le suivant va
    # redemander le même outil : on réutilise le résultat au lieu de le relancer
    # (évite par exemple d'ajouter deux fois la même tâche à la todo-list).
    cache = {}

    def execute_once(name: str, args: dict) -> str:
        key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if key not in cache:
            cache[key] = execute_tool(name, args)
        return cache[key]

    errors = []
    for i, (name, run) in enumerate(_CHAIN):
        try:
            text = run(system, history, latest_message, tools, execute_once)
            return text, name if i == 0 else f"{name} (secours)"
        except Exception as e:
            short = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
            print(f"[Sebastian] {name} indisponible : {short}")
            errors.append(f"{name} : {short}")

    raise RuntimeError("Tous les fournisseurs ont échoué → " + " | ".join(errors))
