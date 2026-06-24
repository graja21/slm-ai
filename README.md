# SLM AI Backend

Backend IA développé dans le cadre du stage chez Value pour explorer et intégrer des Small Language Models (SLMs) exécutés localement avec Ollama.

---

# Objectif

Le projet vise à analyser des documents financiers à l'aide de modèles d'intelligence artificielle locaux sans dépendre d'API cloud externes.

Les objectifs principaux sont :

- Tester plusieurs Small Language Models (SLMs)
- Construire une API REST avec FastAPI
- Réaliser des tâches NLP
- Extraire des informations financières depuis des PDF
- Implémenter un système RAG (Retrieval-Augmented Generation)
- Comparer les performances des différents modèles

---

# Architecture

```text
Frontend (Phase 3)
        |
        v
Spring Boot Bridge (Phase 2)
        |
        v
FastAPI Backend IA
        |
        v
Ollama
        |
        +--> Mistral
        +--> Gemma2
        +--> Phi3
        +--> Llama3.2
```

---

# Technologies utilisées

- Python
- FastAPI
- Ollama
- Mistral
- Gemma2
- Phi3
- Llama3.2
- Sentence Transformers
- PyPDF
- NumPy
- Scikit-Learn
- MLflow
- Git
- GitHub

---

# Fonctionnalités

## NLP

- Classification documentaire
- Résumé automatique
- Extraction d'informations
- Extraction financière structurée

## PDF Processing

- Lecture PDF
- Analyse de rapports financiers
- Découpage en chunks
- Fusion intelligente des résultats
- Validation automatique

## RAG

- Indexation de documents PDF
- Embeddings locaux
- Recherche sémantique
- Recherche par mots-clés
- Hybrid Retrieval
- Stockage persistant

## Monitoring

- Tracking MLflow
- Validation JSON
- Benchmarks

---

# Modèles testés

| Modèle | Statut |
|----------|----------|
| Mistral | Utilisé comme modèle principal |
| Gemma2 | Testé |
| Phi3 | Testé |
| Llama3.2 | Testé |

---

# Résultats des Benchmarks

## Classification documentaire

Tests réalisés sur plusieurs catégories :

- Finance
- Informatique
- Ressources Humaines
- Santé
- Juridique

Résultat :

| Modèle | Performance |
|----------|----------|
| Gemma2 | Meilleur classificateur |
| Mistral | Très bon |
| Phi3 | Correct |
| Llama3.2 | Moyen |

---

## Extraction financière PDF

Résultat global :

| Modèle | Résultat |
|----------|----------|
| Mistral | Meilleure extraction globale |
| Gemma2 | JSON stable mais moins complet |
| Phi3 | Résultats instables |
| Llama3.2 | Résultats incomplets |

Mistral a été retenu comme modèle principal.

---

# Documents testés

- Amen Bank
- BH Bank
- Attijari Premium SICAV

---

# Résultats Financiers Finaux

## Amen Bank

| Champ | Valeur |
|----------|----------|
| Company Name | Amen Bank |
| Document Type | Bilan |
| Period | 31/12/2025 |
| Total Assets | 12 569 041 |
| Net Assets | 1 707 363 |
| Revenue | 1 234 762 |
| Net Profit | 248 652 |
| Validation Status | Valid |

---

## BH Bank

| Champ | Valeur |
|----------|----------|
| Company Name | BH Bank |
| Total Assets | 1 524 878 |
| Net Assets | 1 373 273 |
| Net Profit | 39 769 |
| Validation Status | Valid |

---

# Système RAG

Le projet inclut un système RAG persistant pour répondre à des questions sur des documents PDF.

## Fonctionnement

1. Upload du PDF
2. Extraction du texte
3. Découpage en chunks
4. Création des embeddings
5. Stockage persistant
6. Recherche hybride :
   - Recherche par mots-clés
   - Recherche sémantique
7. Génération de la réponse avec Mistral

---

## Stockage

```text
rag_storage/rag_store.pkl
```

Le document indexé reste disponible après redémarrage du backend.

---

## Exemple

Question :

```text
Quel est le résultat de l'exercice ?
```

Réponse :

```text
248 652 KDT
```

---

# Endpoints Disponibles

## Général

```http
GET /
```

---

## Classification

```http
POST /classify
```

Classification documentaire :

- Finance
- Informatique
- Ressources Humaines
- Santé
- Juridique
- Actualités

---

## Résumé

```http
POST /summarize
```

Résumé automatique de texte.

---

## Extraction

```http
POST /extract
```

Extraction :

- personnes
- entreprises
- lieux
- dates
- nombres
- mots-clés

---

## Extraction Financière

```http
POST /financial-extract
```

Extraction structurée :

- revenus
- résultat net
- dépenses
- devise
- indicateurs financiers

---

## PDF

```http
POST /upload-pdf
```

Résumé PDF.

---

```http
POST /financial-pdf
```

Extraction financière simple.

---

```http
POST /financial-pdf-chunked
```

Extraction financière avancée avec chunking.

---

## RAG

```http
POST /index-pdf
```

Indexation du document.

---

```http
GET /rag-status
```

Statut du stockage RAG.

---

```http
POST /ask-document
```

Question/Réponse sur le document indexé.

---

# Installation

## Cloner le projet

```bash
git clone <repository-url>
```

---

## Se placer dans le dossier

```bash
cd C:\Users\Amine\Desktop\slm-stage\ai-backend
```

---

## Activer l'environnement virtuel

```bash
venv\Scripts\activate
```

---

## Installer les dépendances

```bash
pip install -r requirements.txt
```

---

# Lancer Ollama

```bash
ollama run mistral
```

Modèles testés :

```bash
ollama pull mistral
ollama pull gemma2
ollama pull phi3
ollama pull llama3.2
```

---

# Lancer FastAPI

```bash
uvicorn main:app --reload
```

Swagger :

```text
http://127.0.0.1:8000/docs
```

---

# Lancer MLflow

```bash
mlflow ui
```

Interface :

```text
http://127.0.0.1:5000
```

---

# Résultats du Projet

## Phase 1 Backend & IA

### Réalisé

- Installation Ollama
- Test de plusieurs SLMs
- API REST FastAPI
- Classification documentaire
- Résumé automatique
- Extraction d'informations
- Extraction financière
- Analyse PDF
- Chunking
- Validation JSON
- Tracking MLflow
- RAG persistant
- Hybrid Retrieval
- Benchmarks

### Résultat

Projet fonctionnel permettant :

- l'analyse documentaire
- l'analyse financière
- le question answering sur PDF
- l'exécution locale de modèles IA

sans dépendance à des APIs externes.

---

# Améliorations Futures

- Fine-tuning de modèles
- RAG multi-documents
- Base vectorielle dédiée (FAISS / ChromaDB)
- Interface Angular
- Intégration Spring Boot
- Authentification et sécurisation des APIs
- Dashboard de comparaison des modèles

---

# Auteur

Amine

Stage IA – Value

2026