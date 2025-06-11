# Analýza dát o spotrebe energie a environmentálnych faktoroch

Tento projekt sa zameriava na hĺbkovú analýzu dát a identifikáciu komplexných vzťahov medzi rôznymi premennými. Využíva korelačnú analýzu, teóriu grafov a modely veľkých jazykov (LLM) na extrakciu a interpretáciu poznatkov z dát. Hoci je priložený príklad z oblasti spotreby energie, **architektúra je flexibilná a umožňuje analýzu akéhokoľvek štruktúrovaného datasetu s relevantnými popisnými informáciami pre LLM.**

---

## Štruktúra projektu

Projekt je rozdelený do viacerých Python skriptov, z ktorých každý má špecifickú úlohu v celkovom procese analýzy:

-   `main_analysis.py`: Hlavný skript pre predbežnú analýzu dát, výpočet korelácií, konštrukciu grafu a identifikáciu podgrafov.
-   `main_llm_analysis.py`: Skript, ktorý využíva LLM na analýzu a syntézu zistených podgrafov.
-   `graph_analyzer.py`: Knižnica s funkciami pre prácu s grafmi, vrátane ich konštrukcie, hľadania podgrafov a vizualizácie.
-   `llm_logic.py`: Knižnica s funkciami pre interakciu s LLM, vrátane tvorby promptov a syntézy odpovedí.
-   `data.py`: **Konfiguračný súbor, ktorý obsahuje popisy atribútov vášho konkrétneho datasetu.**

---

## Popis jednotlivých súborov

### `main_analysis.py`

Tento skript je vstupným bodom pre spracovanie surových dát. Jeho hlavné funkcie zahŕňajú:

1.  **Načítanie a príprava dát:** Načíta váš **CSV dataset** (predvolene `datasets/energydata.csv`) a predpripraví ho na analýzu (napr. odstráni nepotrebné stĺpce).
2.  **Výpočet korelačnej matice:** Zistí korelačné vzťahy medzi všetkými numerickými premennými.
3.  **Filtrácia korelačnej matice:** Aplikuje dynamický prah na korelačnú maticu, čím zachová len najvýznamnejšie vzťahy.
4.  **Konštrukcia grafu:** Z prefiltrovanej korelačnej matice vytvorí graf, kde uzly reprezentujú atribúty a hrany ich významné korelácie.
5.  **Nájdenie podgrafov:** Identifikuje špecifické štruktúry v grafe, ako sú **kliky** (skupiny navzájom silne korelovaných atribútov) a **"claw" podgrafy** (jeden centrálny atribút silne koreluje s viacerými inými, ktoré však medzi sebou nekorelujú).
6.  **Vizualizácia grafov:** Generuje a ukladá vizualizácie hlavného grafu, klík a "claw" podgrafov do SVG súborov.
7.  **Uloženie podgrafov pre LLM analýzu:** Exportuje nájdené kliky a "claw" podgrafy do JSON súboru (`subgraphs_to_analyze.json`), ktoré sú následne použité pre LLM analýzu.

### `main_llm_analysis.py`

Tento skript slúži na spustenie LLM analýzy pre podgrafy vygenerované skriptom `main_analysis.py`.

1.  **Načítanie podgrafov:** Načíta podgrafy z `subgraphs_to_analyze.json`.
2.  **LLM Analýza:** Pre každý podgraf zavolá LLM (`llm_logic.py`), ktorý sa pokúsi nájsť a syntetizovať vedecké alebo praktické súvislosti medzi atribútmi v podgrafe, pričom využíva ich popisy z `data.py`.
3.  **Uloženie výsledkov:** Výsledky LLM analýzy, vrátane pôvodných odpovedí a finálnej syntetizovanej odpovede, sú uložené do `llm_analysis_results.json`.

### `graph_analyzer.py`

Táto knižnica obsahuje základné funkcie pre prácu s grafmi, ktoré sú použité v `main_analysis.py`:

-   `construct_graph`: Vytvára `networkx` graf z korelačnej matice.
-   `find_claw_subgraphs`: Identifikuje "claw" podgrafy.
-   `find_cliques`: Nachádza kliky (kompletné podgrafy).
-   `create_main_graph_figure`, `create_claw_subgraphs_figure`, `create_clique_figures`: Funkcie na generovanie Matplotlib figúr pre rôzne vizualizácie grafov.
-   `improved_bridge_layout`, `create_hierarchical_layout`: Pomocné funkcie pre vylepšené rozloženie grafov.

### `llm_logic.py`

Táto knižnica je zodpovedná za interakciu s modelom Llama CPP:

-   **Načítanie modelu:** Pri importe modulu sa načíta predtrénovaný GGUF model. **Cestu k modelu (`MODEL_PATH`) je možné upraviť** tak, aby odkazovala na váš zvolený LLM.
-   `create_analysis_prompt`: Vytvára prompt pre LLM pre počiatočnú analýzu súvislostí medzi atribútmi.
-   `create_synthesis_prompt`: Vytvára "meta-prompt", ktorý inštruuje LLM, aby syntetizoval viaceré predchádzajúce odpovede do jednej koherentnej odpovede.
-   `get_synthesized_answer`: Orchestruje celý proces generovania viacerých odpovedí a ich následnú syntézu pomocou LLM.

### `data.py`

Tento súbor je kľúčový pre správne fungovanie LLM analýzy. Obsahuje slovník `ATTRIBUTE_DESCRIPTIONS`, ktorý poskytuje **textové popisy pre každý atribút/stĺpec vo vašom datasetu.** Tieto popisy sú nevyhnutné pre to, aby LLM mohol správne interpretovať význam premenných a nájsť relevantné súvislosti. **Pri zmene datasetu je nutné aktualizovať tento súbor tak, aby zodpovedal novým atribútom.**

---

## Požiadavky

Pred spustením skriptov sa uistite, že máte nainštalované všetky potrebné knižnice a stiahnutý LLM model.

### Inštalácia knižníc

```bash
pip install pandas numpy networkx matplotlib llama-cpp-python
```

### LLM Model

Stiahnite si kompatibilný GGUF model (napr. `mistral-7b-instruct-v0.2.Q4_K_M.gguf`) a umiestnite ho do adresára `./models/`. Ak adresár `models` neexistuje, vytvorte ho. **Uistite sa, že cesta k modelu v `llm_logic.py` zodpovedá jeho umiestneniu.**

---

## Postup spustenia

Pre plnú analýzu postupujte podľa týchto krokov:

1.  **Pripravte si dataset:**
    Umiestnite váš CSV dataset do adresára `datasets/`. Predvolený názov súboru je `energydata.csv`, ale **môžete ho zmeniť v `main_analysis.py`**.

2.  **Upravte `data.py`:**
    **Veľmi dôležité:** Ak používate iný dataset ako príkladový `energydata.csv`, **upravte slovník `ATTRIBUTE_DESCRIPTIONS` v súbore `data.py`** tak, aby presne popisoval stĺpce vášho datasetu. Toto zabezpečí, že LLM bude rozumieť kontextu vašich dát.

3.  **Spustite `main_analysis.py`:**
    Tento skript vykoná korelačnú analýzu, vytvorí grafy, nájde podgrafy a uloží vizualizácie aj JSON súbor s podgrafmi pre LLM analýzu.

    ```bash
    python main_analysis.py
    ```
    Výstupy (vizualizácie a JSON súbory s klikami a claw podgrafmi) budú uložené do priečinka `outputs/YYYY-MM-DD_HH-MM-SS/`. Súbor `subgraphs_to_analyze.json` pre LLM analýzu bude uložený v `analyzed_subgraphs/`.

4.  **Spustite `main_llm_analysis.py`:**
    Tento skript načíta podgrafy pripravené v predchádzajúcom kroku a spustí LLM analýzu. Výsledky (syntetizované odpovede LLM) budú uložené do `analyzed_subgraphs/llm_analysis_results.json`.

    ```bash
    python main_llm_analysis.py
    ```

---

## Očakávané výstupy

Po úspešnom spustení oboch skriptov nájdete v štruktúre projektu nasledujúce súbory a adresáre:

-   `outputs/YYYY-MM-DD_HH-MM-SS/`: Adresár s vizualizáciami grafov a JSON súbormi klík a "claw" podgrafov.
    -   `main_graph_hierarchical.svg`: Hlavný graf s vizualizáciou všetkých zistených korelácií.
    -   `cliques_size_X.svg`: Vizualizácie klík zoskupených podľa ich veľkosti.
    -   `claw_subgraphs.svg`: Vizualizácia všetkých nájdených "claw" podgrafov.
    -   `cliques.json`: JSON súbor so všetkými nájdenými klikami.
    -   `claws.json`: JSON súbor so všetkými nájdenými "claw" podgrafmi.
-   `analyzed_subgraphs/`: Adresár pre dáta určené pre a z LLM analýzy.
    -   `subgraphs_to_analyze.json`: Podgrafy (kliky a "claws") vybrané pre LLM analýzu.
    -   `llm_analysis_results.json`: Výsledky LLM analýzy pre každý podgraf.

---

## Výstup z LLM Analýzy

Po úspešnom spustení `main_llm_analysis.py` nájdete v priečinku `analyzed_subgraphs/` súbor s názvom `llm_analysis_results.json`. Tento súbor obsahuje detailné výsledky analýzy pre každý podgraf, ktorý bol odovzdaný LLM.

Štruktúra JSON súboru je nasledovná:

```json
[
  {
    "subgraph_nodes": ["T1", "RH_1", "T2"],
    "synthesized_analysis": "Yes, there is a direct link. T1 (Kitchen Temperature) and RH_1 (Kitchen Humidity) are directly linked as they are measurements from the same area. T2 (Living Room Temperature) is also directly linked to T1 and RH_1 due to heat transfer and air circulation between adjacent rooms, meaning changes in one affect the other.",
    "original_responses": [
      "Yes, there is a direct link between T1, RH_1, and T2. T1 and RH_1 are measurements from the same kitchen area, thus directly related. T2 (living room temperature) is related to T1 and RH_1 due to thermal exchange between rooms.",
      "Yes, there is a direct link. T1 and RH_1 are both measurements within the kitchen area, inherently connected. T2, representing the living room temperature, is connected due to the natural flow of heat and air between adjacent living spaces.",
      "Yes, there is a direct link. T1 and RH_1 are directly related as they refer to the temperature and humidity of the same space (kitchen). T2, the living room temperature, is influenced by the kitchen's conditions due to heat transfer and air movement between the rooms."
    ]
  },
  {
    "subgraph_nodes": ["T6", "RH_6", "Wind speed"],
    "synthesized_analysis": "Yes, there is a direct link. All three attributes are outdoor environmental factors. T6 (outdoor temperature), RH_6 (outdoor humidity), and Wind speed are interconnected as wind influences both temperature perception and the rate of moisture evaporation/condensation, thereby affecting humidity.",
    "original_responses": [
      "Yes, there is a direct link. T6 (Temperature outside), RH_6 (Humidity outside), and Wind speed are directly linked as they are all external weather conditions. Wind speed can influence both temperature and humidity.",
      "Yes, there is a direct link. These are all external weather parameters measured outside the building. Wind speed directly affects perceived temperature and can influence the rate of evaporation, thus impacting humidity.",
      "Yes, there is a direct link. T6 and RH_6 are outdoor temperature and humidity, which are influenced by Wind speed, as wind can accelerate heat transfer and moisture dispersion."
    ]
  }
]
```

### Popis štruktúry výstupu

Každý objekt v tomto JSON zozname reprezentuje analýzu jedného podgrafu a obsahuje tri kľúčové polia:

* **`subgraph_nodes`**: Toto pole obsahuje **zoznam názvov uzlov (atribútov)**, ktoré tvorili daný podgraf. Ide o tie isté uzly, ktoré boli odoslané LLM na analýzu.
* **`synthesized_analysis`**: Toto je **finálna, syntetizovaná odpoveď od LLM**. LLM túto odpoveď vytvorilo na základe viacerých pokusov o analýzu (`original_responses`). Je to pokus o najlepšiu a najinformatívnejšiu sumarizáciu vzťahov medzi atribútmi v podgrafe. Začína sa odpoveďou `Yes` alebo `No`, indikujúc, či LLM identifikovalo priamy súvis.
* **`original_responses`**: Toto pole obsahuje **zoznam pôvodných, nezávislých odpovedí**, ktoré LLM vygenerovalo v prvej fáze analýzy. Tieto odpovede boli následne použité ako vstup pre syntézu finálnej odpovede. Počet týchto odpovedí závisí od parametra `retries` v `llm_logic.py` (predvolene 3).

Tento výstup vám poskytne komplexný prehľad o tom, ako LLM interpretovalo vzťahy v jednotlivých podgrafoch, a umožní vám porovnať syntetizované výsledky s pôvodnými odpoveďami.
