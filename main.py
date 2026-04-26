#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PLUTO - Sistema de Inspección de Calidad Industrial
====================================================
Proyecto PIIA - Cliente CTAG - v2.0

Motor    : Experimento 05 (VAE + CatBoost)
Interfaz : Gradio  ->  http://localhost:7860

Uso:
    python main.py

Si el modelo no está generado todavía:
    python scripts/export_exp05_model.py
    python main.py
"""

import os
import sys

# Fix encoding en terminales Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.ui import build_app

if __name__ == "__main__":
    application = build_app()
    application.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
    )

