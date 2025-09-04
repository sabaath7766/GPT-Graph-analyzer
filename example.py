import subprocess
import sys
from graph_analyzer_lib import GraphAnalyzer


def main():
    """
    Príklad použitia knižnice GraphAnalyzer a následné spustenie servera.
    """
    print("=" * 20 + " KROK 1: INICIALIZÁCIA A SPUSTENIE ANALÝZY " + "=" * 20)

    try:
        # Vstup dát a popisov: Cesty k súborom ako vstupy
        analyzer = GraphAnalyzer(
            csv_path="datasets/energydata.csv",
            json_descriptions_path="attribute_descriptions.json",
            alpha=0.1
        )

        # Metóda analyze() sa postará o všetko:
        # - Skontroluje, či výsledok už existuje (perzistencia)
        # - Ak nie, vykoná dátovú a LLM analýzu (paralelne)
        # - Uloží výsledok do results/datasetName-alpha.json
        results = analyzer.analyze()

        if results:
            print("\n✅ Analýza pripravená.")
        else:
            raise Exception("Analýza zlyhala a nevrátila žiadne výsledky.")

    except Exception as e:
        print(f"\n❌ CHYBA počas analýzy: {e}")
        return

    print("\n" + "=" * 20 + " KROK 2: SPUSTENIE SERVERA " + "=" * 20)
    print("Pre zobrazenie výsledkov spúšťam Flask server.")
    print("Po spustení otvor v prehliadači adresu http://127.0.0.1:5000")

    try:
        # Použijeme subprocess, aby sme spustili server v rovnakom prostredí
        subprocess.run([sys.executable, "-m", "flask", "run"])
    except KeyboardInterrupt:
        print("\nServer bol ukončený používateľom.")
    except Exception as e:
        print(f"\n❌ CHYBA pri spúšťaní servera: {e}")


if __name__ == "__main__":
    main()