## Diabetes MLOps Project

Used diabetes dataset from scikit-learn in this project.

### Milestone
- Built and trained an ML model
- Used MLflow for versioning
- Used docker for the inference
- Deployed it on [Diabetes](diabetes.machinelearningdev.com/docs) with persistent storage
- Made it accessible for public

### What’s Inside

| Layer          | Tools Used                                                                 |
|----------------|----------------------------------------------------------------------------|
| **Data**       | `pandas`, `scikit-learn`, `DVC`, `MinIO (S3)`                              |
| **Modeling**   | `scikit-learn (Ridge Regression)`, `MLflow`                                |
| **Tracking**   | `MLflow Tracking & Registry` (self-hosted)                                 |
| **Serving**    | `FastAPI`, `MLflow Model Registry`, `Uvicorn`                              |
| **Monitoring** | `PostgreSQL` (logging, predictions), `SQLAlchemy`, `Alembic`                |
| **UI**         | `Streamlit` dashboard (for exploring predictions)                          |
| **CI/CD**      | `GitHub Actions` for model training and deployment                         |
| **Infrastructure** | `Docker`, `Docker Compose`, deployed via `Coolify`, `Render`           |

### Project Structure

```bash
.
├── app.py                # FastAPI app
├── train.py              # Model training script
├── requirements.txt
├── db.py                 # SQLAlchemy session
├── models.py             # Prediction table
├── alembic/              # DB migrations
├── dashboard/
│   └── app.py            # Streamlit dashboard
├── data/                 # Raw and processed data (tracked via DVC)
├── .dvc/                 # DVC metadata
└── docker-compose.yml    # For API + MLflow setup
```
### Workflow

#### 1. Data Versioning
- Data is stored in CSV format and versioned using **DVC**
- Remote storage is handled via a self-hosted **MinIO S3** bucket

#### 2. Model Training
- Ridge regression model trained on the diabetes dataset
- Parameters, metrics, and artifacts logged to **MLflow**
- Model is **registered** in MLflow with versioning and staging (e.g., `Production`)

#### 3. Serving
- `FastAPI` app loads the latest model from the **MLflow registry**
- Inference logs (input, prediction, model version, timestamp) saved to **PostgreSQL**

#### 4. Migrations
- **Alembic** used for version-controlling database schema
- Fully automated migrations from models

####  5. Monitoring UI
- A lightweight **Streamlit dashboard** connects to PostgreSQL
- Displays recent predictions, inputs, timestamps, and model versions

#### 6. CI/CD
- **GitHub Actions** runs `train.py` on push to `main`
- Model is logged to MLflow, and API automatically reloads the new model


#### Features Completed
- [x] MLflow tracking + registry
- [x] Serving latest Production model
- [x] PostgreSQL inference logging
- [x] CI/CD with GitHub Actions

- [x] Dockerized services (API + MLflow)
- [x] Private MinIO artifact store

####  WIP
- [x] Model version change detection
- [x] Data drift alerts
- [x] Streamlit dashboard for inspection → React frontend
- [x] Authentication for admin dashboard
- [x] Monitoring (Prometheus + Grafana)


#### Author
**Mahesh Sunuwar**
Machine Learning Engineer

Connect: [LinkedIn](https://www.linkedin.com/in/mahesh-sunuwar/) | [GitHub](https://github.com/maheshsunuwar)
