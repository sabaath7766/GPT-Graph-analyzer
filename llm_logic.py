from llama_cpp import Llama
import os

MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model nebol nájdený na ceste: {MODEL_PATH}.")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)


def create_analysis_prompt(subgraph_nodes: list, descriptions: dict, subgraph_type: str) -> str:
    """Creates a context-aware prompt based on the subgraph type (clique or claw)."""
    prompt_parts = [f"- **{node}**: {descriptions.get(node, 'No description available.')}" for node in subgraph_nodes]
    description_text = "\n".join(prompt_parts)

    if subgraph_type == 'clique':
        prompt = f"""
Context: The following attributes form a 'clique', meaning every attribute is strongly correlated with every other attribute in the group.
Attributes:
{description_text}

Question: What is the underlying physical phenomenon, system, or shared environmental factor that best explains this tight, mutual correlation?
Instructions: Provide a concise, one-sentence hypothesis. Start your answer with "Yes, the connection is..." or "No, a direct link is unlikely...".
Answer:"""
    elif subgraph_type == 'claw':
        central_node = subgraph_nodes[0]
        leaf_nodes = ", ".join(subgraph_nodes[1:])
        prompt = f"""
Context: The following attributes form a 'claw' structure. The central attribute is '{central_node}'. The 'leaf' attributes ({leaf_nodes}) are each correlated with the central one, but not with each other.
Attributes:
{description_text}

Question: What is the most likely causal or influential relationship here? How does the central attribute '{central_node}' act as a driver or a common link for the other attributes?
Instructions: Provide a concise, one-sentence explanation of the central attribute's role.
Answer:"""
    else:  # Fallback pre generický prípad
        prompt = f"""
Question: What is the most likely scientific or practical connection between the following attributes?
Attributes:
{description_text}
Instructions: Answer with 'Yes' or 'No', and provide a brief, one-sentence explanation.
Answer:"""

    return prompt


def create_synthesis_prompt(original_question: str, answers: list) -> str:
    answer_text = ""
    for i, ans in enumerate(answers):
        answer_text += f"Answer #{i + 1}:\n\"{ans}\"\n\n"
    prompt = f"""
The original question was: "{original_question}"
I received the following three independent answers from an AI assistant:
{answer_text}
Your task is to carefully analyze these three answers. Identify the common conclusion and formulate one final, best, and most informative answer that combines them and considers all the information provided.
Final Synthesized Answer:"""
    return prompt


def get_synthesized_answer(subgraph_nodes: list, descriptions: dict, subgraph_type: str, retries: int = 3) -> tuple:
    # Krok 1: Generovanie
    initial_prompt = create_analysis_prompt(subgraph_nodes, descriptions, subgraph_type)
    initial_responses = []
    for _ in range(retries):
        output = llm(initial_prompt, max_tokens=100, stop=["\n", "Question:"], echo=False)
        initial_responses.append(output['choices'][0]['text'].strip())

    # Krok 2: Syntéza
    synthesis_prompt = create_synthesis_prompt(
        original_question=f"What is the connection between the attributes {', '.join(subgraph_nodes)}?",
        answers=initial_responses
    )
    final_output = llm(synthesis_prompt, max_tokens=150, stop=["\n"], echo=False)
    final_answer = final_output['choices'][0]['text'].strip()

    return final_answer, initial_responses