# Súbor: run.py
from graph_analyzer_lib import GraphAnalyzer
import subprocess
import sys


def run_analysis_and_server():
    """
    Hlavná funkcia, ktorá demonštruje použitie knižnice GraphAnalyzer
    a následne spustí webový server.
    """
    # --- KROK 1: Konfigurácia vstupov a paralelizácie ---
    csv_file = "datasets/energydata.csv"
    descriptions_file = "datasets/attribute_descriptions.json"
    analysis_alpha = 0.11

    # !!! NASTAVTE PODĽA VAŠEJ RAM/VRAM !!!
    NUM_WORKERS = 2

    print("=" * 20 + " KROK 1: INICIALIZÁCIA ANALYZÁTORA " + "=" * 20)

    try:
        analyzer = GraphAnalyzer(
            csv_path=csv_file,
            descriptions_json_path=descriptions_file,
            alpha=analysis_alpha
        )
    except FileNotFoundError as e:
        print(f"\n CHYBA: {e}")
        return

    # --- KROK 2: Spustenie analýzy ---
    print("\n" + "=" * 20 + " KROK 2: SPUSTENIE ANALÝZY " + "=" * 20)
    try:
        # Odovzdáme počet workerov do metódy analyze
        analyzer.analyze(num_llm_workers=NUM_WORKERS)
    except Exception as e:
        print(f"\n CHYBA počas analýzy: {e}")
        return

    # --- KROK 3: Spustenie servera ---
    print("\n" + "=" * 20 + " KROK 3: SPUSTENIE SERVERA " + "=" * 20)
    try:
        subprocess.run([sys.executable, "server.py"])
    except KeyboardInterrupt:
        print("\nServer bol ukončený používateľom.")
    except Exception as e:
        print(f"\n CHYBA pri spúšťaní servera: {e}")


# --- DÔLEŽITÁ OCHRANA PRE MULTIPROCESSING ---
# Tento blok zabezpečí, že kód sa spustí len vtedy, keď je skript
# priamo vykonaný, a nie keď je importovaný iným procesom.
if __name__ == "__main__":
    # Pre multiprocessing je dobré nastaviť 'spawn' metódu pre lepšiu izoláciu
    # Tento riadok musí byť hneď po if __name__ == "__main__":
    try:
        import multiprocessing as mp

        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # set_start_method sa dá nastaviť len raz

    run_analysis_and_server()
