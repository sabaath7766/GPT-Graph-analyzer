from llama_cpp import Llama
import os

# --- Načítanie modelu ---
# Model sa načíta len raz pri importe modulu, čo šetrí čas a pamäť.
MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model nebol nájdený na ceste: {MODEL_PATH}. Uistite sa, že ste ho stiahli a umiestnili správne.")

print("Načítavam LLM model... (môže to chvíľu trvať)")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,  # Maximálny kontext, ktorý model spracuje
    n_gpu_layers=-1,  # Pokúsi sa použiť GPU, ak je k dispozícii (nastavte na 0, ak chcete použiť iba CPU)
    verbose=False  # Skryje detailné výpisy z llama.cpp
)
print("LLM model úspešne načítaný.")


def create_analysis_prompt(subgraph_nodes: list, descriptions: dict) -> str:
    """Vytvorí úvodný prompt pre analýzu súvislostí v podgrafe."""
    prompt_parts = []
    for node in subgraph_nodes:
        description = descriptions.get(node, 'Popis nie je k dispozícii.')
        prompt_parts.append(f"- **{node}**: {description}")
    description_text = "\n".join(prompt_parts)

    prompt = f"""
Question: What is the most likely scientific or practical connection between the following attributes?
{description_text}
Directions. Start with ‘Yes’ or ‘No’, depending on whether there is a direct link, and then explain why in one or two sentences.
Answer:"""
    return prompt


def create_synthesis_prompt(original_question: str, answers: list) -> str:
    """Vytvorí "meta-prompt", ktorý požiada LLM o syntézu predchádzajúcich odpovedí."""
    answer_text = ""
    for i, ans in enumerate(answers):
        answer_text += f"Odpoveď asistenta #{i + 1}:\n\"{ans}\"\n\n"

    prompt = f"""
The original question was:"{original_question}"

I received the following three independent responses to it from the AI assistant:
{answer_text}
Your task is to carefully analyse these three answers. Identify a common conclusion (e.g. whether a link exists) and 
formulate one final, best and most informative answer that brings them together and takes into account all the 
information provided. Start with ‘Yes’ or ‘No’.

Final synthesised answer:"""
    return prompt


def get_synthesized_answer(subgraph_nodes: list, descriptions: dict, retries: int = 3) -> tuple:
    """
    Orchestruje celý proces: generovanie viacerých odpovedí a ich následnú syntézu.

    Returns:
        tuple: Obsahuje (finálna_odpoveď, zoznam_pôvodných_odpovedí)
    """
    # --- Fáza 1: Generovanie ---
    initial_prompt = create_analysis_prompt(subgraph_nodes, descriptions)
    initial_responses = []
    for _ in range(retries):
        output = llm(initial_prompt, max_tokens=100, stop=["\n", "Otázka:"], echo=False)
        initial_responses.append(output['choices'][0]['text'].strip())

    # --- Fáza 2: Syntéza ---
    synthesis_prompt = create_synthesis_prompt(
        original_question=f"What is the relationship between the attributes {', '.join(subgraph_nodes)}?",
        answers=initial_responses
    )

    final_output = llm(synthesis_prompt, max_tokens=3072, stop=None, echo=False)
    final_answer = final_output['choices'][0]['text'].strip()

    return final_answer, initial_responses