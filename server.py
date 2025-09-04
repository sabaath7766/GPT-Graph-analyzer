from flask import Flask, render_template, abort
import os
import json
import networkx as nx

# Pomocný modul pre vizualizácie
import graph_analyzer

app = Flask(__name__)
RESULTS_DIR = "results"

@app.route('/')
def index():
    """Zobrazí zoznam dostupných JSON analýz v priečinku /results."""
    try:
        files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]
        return render_template('index.html', files=sorted(files))
    except FileNotFoundError:
        return "Priečinok 'results' nebol nájdený.", 404

@app.route('/analysis/<filename>')
def analysis_detail(filename):
    """Načíta a zobrazí detailnú analýzu z konkrétneho JSON súboru."""
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        return abort(404)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Dynamické generovanie vizualizácií pre jednotlivé podgrafy
    # (Toto sa nerobí vopred, ale až pri zobrazení, aby JSON súbory zostali čisté)
    # V tomto príklade to preskočíme, aby sme nekomplikovali kód,
    # ale pridáme placeholder. V reálnej aplikácii by tu bol kód na generovanie.
    for analysis in data.get("llm_analyses", []):
        # Na jednoduchosť pridáme placeholder, v praxi by sme tu volali
        # graph_analyzer.create_single_clique_viz_base64 atď.
        analysis["visualization_base64"] = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" # Priehľadný pixel

    return render_template('analysis_detail.html', data=data, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)