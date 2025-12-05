# Nikudiboi - AI Hebrew Diacritization System

This folder contains the production-ready deployment version of D-Nikud (rebranded as Nikudiboi).

## 🚀 Quick Start (Local)

1. **Run the server:**
   Double click `start_nikudiboi.bat` in the project root.

2. **Access the App:**
   - **Main Interface:** http://127.0.0.1:8000
   - **Admin Panel:** http://127.0.0.1:8000/admin

## 🐳 Deployment (Docker)

This folder includes a `Dockerfile` optimized for deployment on any container platform (Google Cloud Run, AWS ECS, DigitalOcean App Platform, etc.).

### Build & Run
```bash
# From the PROJECT ROOT directory (D_Nikud)
docker build -f nikudiboi_deploy/Dockerfile -t nikudiboi .

# Run the container
docker run -p 8000:8000 nikudiboi
```

## ⚙️ Configuration
All configuration is managed via the Admin Panel or by editing `site_config.json` (created automatically on first run).

- **Models:** Upload `.pth` files via Admin or place them manually in `nikudiboi_deploy/models/`.
- **UI Customization:** Change title, colors, and texts via Admin.

## 📂 Structure
- `app/` - Application code (FastAPI + HTML Templates).
- `models/` - Directory for storing model weights.
- `Dockerfile` - Instructions for building the container image.
