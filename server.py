# Súbor: server.py

import json
import io
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, abort, Response, request
from flask_cors import CORS
from PIL import Image


import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

RESULTS_DIR = Path("results")


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


# In-memory state — resets on server restart but persists across page reloads
_state = {}

@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify(_state)


@app.route('/api/state', methods=['POST'])
def set_state():
    _state.update(request.json or {})
    return jsonify(_state)


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


@app.route('/api/analysis/<string:filename>')
def get_analysis_data(filename):
    if '..' in filename or filename.startswith('/'):
        abort(400, "Invalid filename.")

    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        abort(404, f"Analysis file '{filename}' not found.")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = data.get('metadata', {})
        if 'correlation_threshold' not in metadata:
            print(f"Info: 'correlation_threshold' chýba v súbore '{filename}'. Dopočítavam...")

            alpha_norm = metadata.get('alpha_normalized')
            csv_path = metadata.get('csv_path')

            if alpha_norm is not None and csv_path and Path(csv_path).exists():
                df = pd.read_csv(csv_path)
                df = df.select_dtypes(include=np.number)
                if df.columns[0].lower() in ['date', 'time', 'unnamed: 0']:
                    df = df.drop(columns=df.columns[0])

                correlation_matrix = df.corr()
                cor_values = correlation_matrix.values[np.where(~np.eye(correlation_matrix.shape[0], dtype=bool))]
                alpha_internal = alpha_norm * 0.3
                threshold = (np.max(cor_values) + np.mean(cor_values)) / 2 + alpha_internal

                data['metadata']['correlation_threshold'] = threshold
                print(f"Info: Prah dopočítaný na hodnotu {threshold:.4f}")
            else:
                print(f"Warning: Nepodarilo sa dopočítať prah pre '{filename}' - chýba alpha alebo cesta k CSV.")

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": f"Failed to read or process analysis file: {e}"}), 500


@app.route('/api/screenshot', methods=['POST'])
def take_screenshot():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return jsonify({"error": "playwright not installed. Run: pip install playwright && playwright install chromium"}), 500

    data = request.json or {}
    scale = data.get('scale', 10)
    dark  = data.get('dark', True)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=['--disable-gpu', '--disable-software-rasterizer'])
            context = browser.new_context(
                viewport={'width': 1440, 'height': 1350},
                device_scale_factor=8
            )
            page = context.new_page()

            if dark:
                context.add_init_script("""
                    Object.defineProperty(window, 'matchMedia', {
                        writable: true,
                        value: (query) => ({
                            matches: query.includes('dark'),
                            media: query,
                            onchange: null,
                            addListener: () => {},
                            removeListener: () => {},
                            addEventListener: () => {},
                            removeEventListener: () => {},
                            dispatchEvent: () => false,
                        }),
                    });
                """)

            page.goto('http://127.0.0.1:5000', wait_until='networkidle')

            if dark:
                page.evaluate("document.documentElement.classList.add('dark')")

            page.wait_for_selector('main .grid .cursor-pointer', timeout=30000)
            page.wait_for_timeout(4000)

            screenshot_bytes = page.screenshot(
                full_page=False,
                clip={'x': 0, 'y': 0, 'width': 1440, 'height': 1350}
            )
            browser.close()

        img = Image.open(io.BytesIO(screenshot_bytes))
        img_scaled = img.resize((1440 * scale, 1350 * scale), Image.LANCZOS)
        output = io.BytesIO()
        img_scaled.save(output, format='PNG')
        output.seek(0)

        return Response(output.read(), mimetype='image/png', headers={
            'Content-Disposition': 'attachment; filename=page_screenshot.png'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("Spúšťam Flask server...")
    print(f"Výsledky sa budú načítavať z priečinka: '{RESULTS_DIR.resolve()}'")
    print("Po spustení servera otvorte v prehliadači súbor 'index.html' alebo adresu http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True)