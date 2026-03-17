# Súbor: run_structural_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from graph_analyzer_lib import GraphAnalyzer


def generate_analysis_plot():
    """
    Spustí štrukturálnu analýzu pre viacero datasetov. Pre každý najprv vygeneruje
    korelačnú heatmapu a následne graf závislosti počtu nájdených štruktúr
    od parametra alpha.
    """
    datasets_to_analyze = {
        "Abalone": (
            "datasets/abalone.csv",
            "datasets/abalone_descriptions.json"
        ),
        "EnergyData": (
            "datasets/energydata.csv",
            "datasets/attribute_descriptions.json"
        ),
        "Superconductivity": (
            "datasets/train.csv",
            "datasets/superconductor_descriptions.json"
        )
    }

    alpha_range = np.arange(0.0, 1.01, 0.01)

    print("=" * 50)
    print("Spúšťam experimentálnu štrukturálnu analýzu...")
    print("=" * 50)

    for dataset_name, (csv_path, desc_path) in datasets_to_analyze.items():
        print(f"\nSpracovávam dataset: '{dataset_name}'")

        # =========================================================================
        # UPRAVENÁ SEKCIA: GENERUJ A ULOŽ FILTROVANÚ HEATMAPU (alpha=1.0)
        # =========================================================================
        print("  Generujem filtrovanú korelačnú heatmapu pre alpha = 1.0...")
        try:
            temp_analyzer = GraphAnalyzer(csv_path=csv_path, descriptions_json_path=desc_path, alpha_normalized=1.0)
            temp_analyzer._load_and_prepare_data()
            df_for_heatmap = temp_analyzer.df
            corr_matrix = df_for_heatmap.corr()

            # --- VÝPOČET A APLIKÁCIA FILTRA ---
            # 1. Definujeme alpha a vypočítame dynamický prah presne ako v knižnici
            ALPHA_FOR_HEATMAP = 0.1
            alpha_internal = ALPHA_FOR_HEATMAP * 0.3
            cor_values = corr_matrix.values[np.where(~np.eye(corr_matrix.shape[0], dtype=bool))]
            threshold = (np.max(cor_values) + np.mean(cor_values)) / 2 + alpha_internal
            print(f"    Vypočítaný prah pre alpha={ALPHA_FOR_HEATMAP:.1f} je: {threshold:.4f}")

            # 2. Aplikujeme filter na maticu
            filtered_corr_matrix_np = np.where(np.abs(corr_matrix) > threshold, corr_matrix, 0)
            np.fill_diagonal(filtered_corr_matrix_np, 0)  # Prečistíme diagonálu
            filtered_corr_matrix = pd.DataFrame(filtered_corr_matrix_np, index=corr_matrix.index,
                                                columns=corr_matrix.columns)

            # --- VYKRESLENIE FILTROVANEJ MATICE ---
            plt.figure(figsize=(20, 18))
            sns.heatmap(filtered_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f'Filtrovaná Korelačná Heatmapa (Alpha = {ALPHA_FOR_HEATMAP:.1f})\nDataset: {dataset_name}',
                      fontsize=20)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout(pad=3.0)

            heatmap_filename = f"heatmap_filtered_alpha_{ALPHA_FOR_HEATMAP:.1f}_{dataset_name}.png"
            plt.savefig(heatmap_filename, dpi=150)
            print(f"  Heatmapa úspešne uložená do súboru: '{heatmap_filename}'")
            plt.close()

        except Exception as e:
            print(f"  CHYBA pri generovaní heatmapy: {e}")

        # Pôvodná analýza vplyvu alpha pokračuje bez zmeny
        results = []
        for alpha in tqdm(alpha_range, desc=f"Analyzujem {dataset_name}"):
            try:
                analyzer = GraphAnalyzer(
                    csv_path=csv_path,
                    descriptions_json_path=desc_path,
                    alpha_normalized=alpha
                )
                num_ptychy, num_hotspoty = analyzer.get_structure_counts()
                results.append({
                    'alpha': alpha,
                    'ptychy': num_ptychy,
                    'hotspoty': num_hotspoty,
                    'total': num_ptychy + num_hotspoty
                })
            except Exception as e:
                print(f"Chyba pri alpha={alpha:.2f} pre dataset {dataset_name}: {e}")
                results.append({'alpha': alpha, 'ptychy': 0, 'hotspoty': 0, 'total': 0})

        if not results:
            print(f"Pre dataset {dataset_name} neboli získané žiadne výsledky pre graf vplyvu alpha.")
            continue

        df = pd.DataFrame(results)

        # =========================================================================
        # NOVÁ SEKCIA: VÝPIS SÚHRNNÝCH ŠTATISTÍK
        # =========================================================================
        print("\n" + "~" * 40)
        print(f"📊 SÚHRNNÉ ŠTATISTIKY PRE {dataset_name.upper()}")
        print("~" * 40)

        # Nájdeme riadok s maximálnym počtom štruktúr
        max_total_row = df.loc[df['total'].idxmax()]
        # Nájdeme riadok s maximálnym počtom n-ptých
        max_ptychy_row = df.loc[df['ptychy'].idxmax()]
        # Nájdeme riadok s maximálnym počtom hotspotov
        max_hotspoty_row = df.loc[df['hotspoty'].idxmax()]

        # Základné štatistiky
        print("▶ Počet n-ptých (Ptychy):")
        print(f"  - Max počet: {df['ptychy'].max()} (pri alpha={max_ptychy_row['alpha']:.2f})")
        print(f"  - Min počet: {df['ptychy'].min()} (pri alpha={df.loc[df['ptychy'].idxmin()]['alpha']:.2f})")
        print(f"  - Priemer: {df['ptychy'].mean():.2f}")

        print("\n▶ Počet Hotspotov:")
        print(f"  - Max počet: {df['hotspoty'].max()} (pri alpha={max_hotspoty_row['alpha']:.2f})")
        print(f"  - Min počet: {df['hotspoty'].min()} (pri alpha={df.loc[df['hotspoty'].idxmin()]['alpha']:.2f})")
        print(f"  - Priemer: {df['hotspoty'].mean():.2f}")

        print("\n▶ Celkový počet štruktúr (Ptychy + Hotspoty):")
        print(f"  - Max celkový počet: {df['total'].max()} (pri alpha={max_total_row['alpha']:.2f})")
        print(f"  - Min celkový počet: {df['total'].min()} (pri alpha={df.loc[df['total'].idxmin()]['alpha']:.2f})")
        print(f"  - Priemer: {df['total'].mean():.2f}")
        print("-" * 40)
        # =========================================================================
        # KONIEC NOVEJ SEKCIE
        # =========================================================================


        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()
        color1, color2 = 'royalblue', 'crimson'
        line1 = ax1.plot(df['alpha'], df['ptychy'], color=color1, label='Počet n-ptých (ľavá os)')
        ax1.set_xlabel('Hodnota Alpha (normalizovaná)', fontsize=12)
        ax1.set_ylabel('Počet n-ptých', color=color1, fontsize=14, weight='bold')
        ax1.tick_params(axis='y', labelcolor=color1)
        line2 = ax2.plot(df['alpha'], df['hotspoty'], color=color2, linestyle='--', label='Počet hotspotov (pravá os)')
        ax2.set_ylabel('Počet hotspotov', color=color2, fontsize=14, weight='bold')
        ax2.tick_params(axis='y', labelcolor=color2)
        max_ptychy = df['ptychy'].max()
        if max_ptychy > 0: ax1.set_ylim(0, max_ptychy * 1.05)
        max_hotspoty = df['hotspoty'].max()
        if max_hotspoty > 0: ax2.set_ylim(0, max_hotspoty * 1.05)
        lines, labels = line1 + line2, [l.get_label() for l in line1 + line2]
        ax1.legend(lines, labels, loc='upper right')
        fig.suptitle(f'Vplyv Alpha na n-ptychy vs. hotspoty\nDataset: {dataset_name}', fontsize=16, weight='bold')
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        output_filename = f"analyza_vplyvu_alpha_dual_axis_{dataset_name}.png"
        plt.savefig(output_filename, dpi=150)
        print(f"Graf vplyvu alpha bol úspešne uložený do súboru: '{output_filename}'")
        plt.show()


if __name__ == '__main__':
    try:
        import multiprocessing as mp

        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    generate_analysis_plot()
