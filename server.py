# server.py

from flask import Flask, jsonify, send_from_directory, abort
from flask_cors import CORS
import os
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Povolí Cross-Origin Resource Sharing pre jednoduchý vývoj

RESULTS_DIR = Path("results")


@app.route('/')
def index():
    """Servuje hlavnú stránku index.html."""
    return send_from_directory('.', 'index.html')


@app.route('/api/analyses')
def list_analyses():
    """Vráti zoznam dostupných JSON súborov s analýzami v priečinku 'results'."""
    if not RESULTS_DIR.exists():
        return jsonify([])

    try:
        json_files = sorted(
            [f.name for f in RESULTS_DIR.iterdir() if f.is_file() and f.suffix == '.json'],
            reverse=True  # Najnovšie analýzy budú prvé
        )
        return jsonify(json_files)
    except Exception as e:
        return jsonify({"error": f"Failed to list analysis files: {e}"}), 500


@app.route('/api/analysis/<string:filename>')
def get_analysis_data(filename):
    """
    Načíta, za behu opraví nekompletné dáta a vráti obsah
    konkrétneho JSON súboru s analýzou.
    """
    if '..' in filename or filename.startswith('/'):
        abort(400, "Invalid filename.")

    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        abort(404, f"Analysis file '{filename}' not found.")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # === ZAČIATOK KĽÚČOVEJ ZMENY: ON-THE-FLY OPRAVA DÁT ===

        # 1. Získame všetky unikátne atribúty, ktoré boli reálne analyzované
        all_analyzed_attributes = set()
        if 'analyses' in data:
            for analysis in data['analyses']:
                if 'attributes' in analysis:
                    all_analyzed_attributes.update(analysis['attributes'].keys())

        # 2. Získame popisy, ktoré sú uložené v súbore (môžu byť nekompletné)
        existing_descriptions = data.get('metadata', {}).get('attribute_descriptions', {})

        # 3. Vytvoríme nový, kompletný slovník popisov
        complete_descriptions = {}
        for attr_name in sorted(list(all_analyzed_attributes)):
            # Použijeme existujúci popis, alebo priradíme predvolený, ak chýba
            complete_descriptions[attr_name] = existing_descriptions.get(
                attr_name,
                "Description not available in JSON file."
            )

        if 'metadata' not in data:
            data['metadata'] = {}
        data['metadata']['attribute_descriptions'] = complete_descriptions

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": f"Failed to read, parse, or patch analysis file: {e}"}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("Spúšťam Flask server...")
    print(f"Výsledky sa budú načítavať z priečinka: '{RESULTS_DIR.resolve()}'")
    print("Po spustení servera otvorte v prehliadači súbor 'index.html' alebo adresu http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True)