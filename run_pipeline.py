# run_pipeline.py

import main_analysis
import main_llm_analysis
import subprocess
import sys

def run_full_pipeline():
    """
    Spustí celý proces:
    1. Analýza dát a vytvorenie podgrafov.
    2. Analýza podgrafov pomocou LLM.
    3. Spustenie Flask servera na zobrazenie výsledkov.
    """
    print("="*20 + " KROK 1: SPUSTENIE DÁTOVEJ ANALÝZY " + "="*20)
    try:
        main_analysis.main()
        print("\n✅ Dátová analýza úspešne dokončená.")
    except Exception as e:
        print(f"\n❌ CHYBA počas dátovej analýzy: {e}")
        return # Ukončíme, ak zlyhá prvý krok

    print("\n" + "="*20 + " KROK 2: SPUSTENIE LLM ANALÝZY " + "="*20)
    try:
        main_llm_analysis.run_llm_analysis()
        print("\n✅ LLM analýza úspešne dokončená.")
    except Exception as e:
        print(f"\n❌ CHYBA počas LLM analýzy: {e}")
        return # Ukončíme, ak zlyhá druhý krok

    print("\n" + "="*20 + " KROK 3: SPUSTENIE SERVERA " + "="*20)
    print("Všetky analýzy sú hotové. Spúšťam Flask server...")
    print("Po spustení servera otvor súbor 'index.html' v prehliadači.")

    try:
        # Použijeme subprocess, aby sme spustili server v rovnakom prostredí
        # a videli jeho výstup priamo v tomto termináli.
        subprocess.run([sys.executable, "server.py"])
    except KeyboardInterrupt:
        print("\nServer bol ukončený používateľom.")
    except Exception as e:
        print(f"\n❌ CHYBA pri spúšťaní servera: {e}")

if __name__ == "__main__":
    run_full_pipeline()