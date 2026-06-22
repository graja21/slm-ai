\# SLM AI Backend



Backend IA développé dans le cadre du stage Value pour explorer les Small Language Models (SLMs) exécutés localement avec Ollama.



\## Objectif



Le projet vise à analyser des documents financiers à l’aide de modèles locaux sans dépendre d’API cloud externes.



\## Technologies utilisées



\- Python

\- FastAPI

\- Ollama

\- Mistral

\- Gemma2

\- Phi3

\- Llama3.2

\- PyPDF

\- MLflow

\- Git / GitHub



\## Fonctionnalités



\- Résumé de documents PDF

\- Classification documentaire

\- Extraction financière structurée

\- Analyse de rapports financiers PDF

\- Chunking des grands documents

\- Fusion des résultats extraits

\- Validation automatique des données

\- Suivi des expériences avec MLflow

\- Comparaison de plusieurs SLMs



\## Modèles testés



\- Mistral

\- Gemma2

\- Phi3

\- Llama3.2



\## Résultats des benchmarks



\### Classification documentaire



Gemma2 a obtenu les meilleurs résultats sur les catégories testées :

Finance, Ressources Humaines, Santé et Juridique.



\### Extraction financière PDF



Mistral a été retenu comme meilleur modèle global grâce à des extractions plus complètes sur les documents financiers réels.



| Modèle | Résultat |

|---|---|

| Mistral | Meilleure extraction globale |

| Gemma2 | JSON stable mais extraction moins complète |

| Phi3 | Résultats instables |

| Llama3.2 | Résultats incomplets |



\## Documents testés



\- BH Bank

\- Amen Bank

\- Attijari Premium SICAV



\## Endpoints principaux



\- `GET /`

\- `POST /classify`

\- `POST /financial-extract`

\- `POST /upload-pdf`

\- `POST /financial-pdf`

\- `POST /financial-pdf-chunked`



\## Lancer le projet



```bash

cd C:\\Users\\Amine\\Desktop\\slm-stage\\ai-backend

venv\\Scripts\\activate

uvicorn main:app --reload


Swagger :
http://127.0.0.1:8000/docs

Lancer Ollama:
ollama run mistral

Lancer MLflow:
mlflow ui
http://127.0.0.1:5000

