# Osteoporosis AI Diagnosis & Clinical Analytics Dashboard

AI-powered osteoporosis diagnostic Streamlit application utilizing a Dual-Branch ResNet-50 CBAM + ViT-B/16 architecture.

## Folder Structure
- `app.py`: Streamlit entry point.
- `train_dual_cbam_vit_optimized.py`: Neural network module architecture.
- `patient_records.csv`: Patient record log database.
- `checkpoints/best_model.pth`: Trained model weights (Git LFS).
- `requirements.txt`: Python package dependencies.
- `.gitattributes`: Git LFS file tracking.

## Local Test Run
```bash
streamlit run app.py
```
