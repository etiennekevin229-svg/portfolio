# Portfolio de Kevin Étienne + Sebastian

Portfolio personnel de **Caïto Kevin Marley Étienne**, développeur full-stack et
ingénieur ERP (Odoo 19), co-fondateur de Proaktif Consulting à Port-au-Prince, Haïti.

Le site intègre **Sebastian**, mon assistant IA, avec lequel les visiteurs peuvent discuter
pour en savoir plus sur mon parcours, mes compétences et mes projets.

## Architecture

```
index.html            Portfolio (HTML / CSS / JavaScript, sans framework)
                      + bulle de chat Sebastian
api/chat.py           Fonction Python serverless (Vercel) : POST /api/chat
lib/llm_router.py     Chaîne de modèles gratuits avec bascule automatique
lib/sebastian_public.py  Instructions et profil public donnés au modèle
data/kevin_public.md  Profil public (seule source de connaissances de Sebastian ici)
```

- **Front-end** : le navigateur envoie la conversation à `/api/chat` et affiche la réponse.
- **Back-end** : une fonction Python serverless sur Vercel, sans framework.
- **Modèles** : Google Gemini → Gemini Flash-Lite → GitHub Models. Si un modèle atteint
  son quota ou tombe en panne, le suivant prend le relais automatiquement.
- **Sécurité** : clés API uniquement dans les variables d'environnement Vercel, aucun outil
  actif côté public, contrôle de l'origine des requêtes, limite de débit par visiteur,
  taille des messages bornée, règles contre la manipulation du modèle, rendu des réponses
  échappé côté navigateur.

La version privée de Sebastian (sur mon PC) ajoute une mémoire persistante, une recherche
dans mes documents (RAG avec ChromaDB), une todo-list et la recherche web.

## Variables d'environnement (Vercel)

| Nom | Rôle |
|---|---|
| `GEMINI_API_KEY` | Clé Google AI Studio (modèle principal) |
| `GITHUB_TOKEN` | Jeton GitHub avec la permission *Models: read* (secours) |
| `GEMINI_MODEL`, `GEMINI_BACKUP_MODEL`, `GITHUB_MODEL` | Facultatif : changer de modèle |
| `ALLOWED_ORIGINS` | Facultatif : autres domaines autorisés à appeler l'API |
