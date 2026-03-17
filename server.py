# Súbor: server.py

from flask import Flask, jsonify, send_from_directory, abort
from flask_cors import CORS
import os
import json
from pathlib import Path

# === ZMENA: Pridanie potrebných importov ===
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

RESULTS_DIR = Path("results")


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/analyses')
def list_analyses():
    if not RESULTS_DIR.exists():
        return jsonify([])
    try:
        json_files = sorted(
            [f.name for f in RESULTS_DIR.iterdir() if f.is_file() and f.suffix == '.json'],
            reverse=True
        )
        return jsonify(json_files)
    except Exception as e:
        return jsonify({"error": f"Failed to list analysis files: {e}"}), 500


# === ZMENA: Nahradenie celej tejto funkcie ===
@app.route('/api/analysis/<string:filename>')
def get_analysis_data(filename):
    """
    Načíta dáta z JSON súboru a ak v nich chýba hodnota prahu,
    dopočíta ju za behu.
    """
    if '..' in filename or filename.startswith('/'):
        abort(400, "Invalid filename.")

    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        abort(404, f"Analysis file '{filename}' not found.")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # --- ON-THE-FLY VÝPOČET PRAHU (ak chýba) ---
        metadata = data.get('metadata', {})
        if 'correlation_threshold' not in metadata:
            print(f"Info: 'correlation_threshold' chýba v súbore '{filename}'. Dopočítavam...")

            alpha_norm = metadata.get('alpha_normalized')
            csv_path = metadata.get('csv_path')

            if alpha_norm is not None and csv_path and Path(csv_path).exists():
                # Načítame a pripravíme dáta presne ako v analytickej knižnici
                df = pd.read_csv(csv_path)
                df = df.select_dtypes(include=np.number)
                if df.columns[0].lower() in ['date', 'time', 'unnamed: 0']:
                    df = df.drop(columns=df.columns[0])

                # Vypočítame prah presne rovnakým vzorcom
                correlation_matrix = df.corr()
                cor_values = correlation_matrix.values[np.where(~np.eye(correlation_matrix.shape[0], dtype=bool))]
                alpha_internal = alpha_norm * 0.3
                threshold = (np.max(cor_values) + np.mean(cor_values)) / 2 + alpha_internal

                # Pridáme dopočítanú hodnotu do dát, ktoré pošleme
                data['metadata']['correlation_threshold'] = threshold
                print(f"Info: Prah dopočítaný na hodnotu {threshold:.4f}")
            else:
                print(f"Warning: Nepodarilo sa dopočítať prah pre '{filename}' - chýba alpha alebo cesta k CSV.")

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": f"Failed to read or process analysis file: {e}"}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("Spúšťam Flask server...")
    print(f"Výsledky sa budú načítavať z priečinka: '{RESULTS_DIR.resolve()}'")
    print("Po spustení servera otvorte v prehliadači súbor 'index.html' alebo adresu http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True)