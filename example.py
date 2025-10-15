# Súbor: example.py
from graph_analyzer_lib import GraphAnalyzer
import subprocess
import sys
import numpy as np  # Importujeme numpy pre jednoduché generovanie rozsahu


def run_batch_analysis_and_server():
    """
    Hlavná funkcia, ktorá najprv spustí dávkovú analýzu pre rôzne hodnoty alfa
    a následne spustí webový server na prehliadanie výsledkov.
    """
    # 1: Konfigurácia vstupov a paralelizácie
    csv_file = "datasets/abalone.csv"
    descriptions_file = "datasets/abalone_descriptions.json"
    NUM_WORKERS = 2

    # timeout for testing
    # TIMEOUT_MINUTES = 5.0

    # Vytvoríme pole hodnôt pre NORMALIZED_ALPHA od 0.1 do 1.0
    alpha_range = np.arange(0.1, 1.1, 0.1)

    # 2: Spustenie dávkovej analýzy v slučke
    print("=" * 20 + " KROK 1 & 2: SPUSTENIE DÁVKOVEJ ANALÝZY " + "=" * 20)
    print(f"Bude vykonaná analýza pre nasledujúce hodnoty alpha: {[round(a, 1) for a in alpha_range]}")

    for alpha_value in alpha_range:
        current_alpha = round(alpha_value, 1)
        print("\n" + "-" * 15 + f" SPÚŠŤAM ANALÝZU PRE ALPHA = {current_alpha} " + "-" * 15)

        try:
            # Inicializácia analyzátora musí byť vnútri slučky,
            # pretože pre každú analýzu používame inú hodnotu alpha.
            analyzer = GraphAnalyzer(
                csv_path=csv_file,
                descriptions_json_path=descriptions_file,
                alpha_normalized=current_alpha
            )

            # Spustenie konkrétnej analýzy
            analyzer.analyze(
                force_reanalyze=True,  # Zabezpečí, že sa analýza vždy vykoná nanovo
                num_llm_workers=NUM_WORKERS
                # timeout_minutes je zakomentovaný, takže ho nepoužívame
            )
            print(f"✅ Analýza pre alpha={current_alpha} úspešne dokončená.")

        except FileNotFoundError as e:
            # Ak chýbajú vstupné súbory, nemá zmysel pokračovať
            print(f"\n❌ KRITICKÁ CHYBA: {e}")
            print("Dávková analýza bola prerušená, pretože chýba dôležitý súbor.")
            return
        except Exception as e:
            # Ak zlyhá jedna analýza, zaznamenáme chybu a pokračujeme ďalšou
            print(f"\n❌ CHYBA počas analýzy pre alpha={current_alpha}: {e}")
            print("Pokračujem na ďalšiu hodnotu alpha...")
            continue

    print("\n" + "=" * 20 + " VŠETKY ANALÝZY BOLI DOKONČENÉ " + "=" * 20)

    # 3: Spustenie servera po dokončení všetkých analýz
    print("\n" + "=" * 20 + " KROK 3: SPUSTENIE SERVERA " + "=" * 20)
    try:
        subprocess.run([sys.executable, "server.py"])
    except KeyboardInterrupt:
        print("\nServer bol ukončený používateľom.")
    except Exception as e:
        print(f"\n❌ CHYBA pri spúšťaní servera: {e}")


# OCHRANA PRE MULTIPROCESSING
# Tento blok zabezpečí, že kód sa spustí len vtedy, keď je skript
# priamo vykonaný, a nie keď je importovaný iným procesom.
if __name__ == "__main__":
    # Pre multiprocessing je dobré nastaviť 'spawn' metódu pre lepšiu izoláciu
    try:
        import multiprocessing as mp

        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Názov funkcie bol zmenený pre lepšiu prehľadnosť
    run_batch_analysis_and_server()