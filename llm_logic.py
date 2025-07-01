# Súbor: llm_logic.py (Verzia s finálnou opravou)
from llama_cpp import Llama
import os

# Načítanie modelu zostáva bez zmeny
MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model nebol nájdený na ceste: {MODEL_PATH}.")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)


def create_analysis_prompt(subgraph_nodes: list, descriptions: dict, subgraph_type: str) -> str:
    # Táto funkcia je už v poriadku a zostáva bez zmeny
    prompt_parts = [f"- **{node}**: {descriptions.get(node, 'No description available.')}" for node in subgraph_nodes]
    description_text = "\n".join(prompt_parts)
    if subgraph_type == 'clique':
        context_description = "The following attributes exhibit a special relationship: every attribute in this group is strongly and directly correlated with every other attribute."
        question = "Is there a clear underlying physical phenomenon or shared system that explains this tight, mutual correlation?"
        instructions = "You MUST start your answer with 'Yes,' followed by a concise, one-sentence hypothesis, or 'No,' if a direct link is unlikely."
    elif subgraph_type == 'claw':
        central_node, leaf_nodes = subgraph_nodes[0], subgraph_nodes[1:]
        context_description = f"The following attributes exhibit a 'hub-and-spoke' relationship. The central attribute '{central_node}' is strongly correlated with each of the 'leaf' attributes: {', '.join(leaf_nodes)}. However, the leaf attributes are not strongly correlated with each other."
        question = f"Is there a clear causal or influential relationship here, with '{central_node}' acting as a driver?"
        instructions = f"You MUST start your answer with 'Yes,' and then explain the role of the central node '{central_node}', or 'No,' if the relationship is not clear."
    else:
        context_description = "Analyze the relationship between the following attributes."
        question = "Is there a likely connection?"
        instructions = "You MUST start your answer with 'Yes,' or 'No,' followed by a brief explanation."
    prompt = f"""
Context: {context_description}
Attributes:
{description_text}
Question: {question}
Instructions: {instructions}
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
Your task: Synthesize these three answers into a single, conclusive, and well-formulated final answer. Identify the common conclusion (especially the 'Yes' or 'No' part) and combine the explanations. Start your response directly with the synthesized conclusion.
Final Synthesized Conclusion:"""
    return prompt


def get_synthesized_answer(subgraph_nodes: list, descriptions: dict, subgraph_type: str, retries: int = 3) -> tuple:
    initial_prompt = create_analysis_prompt(subgraph_nodes, descriptions, subgraph_type)
    initial_responses = []

    for _ in range(retries):
        output = llm(initial_prompt, max_tokens=1024, echo=False)
        response_text = output['choices'][0]['text'].strip()
        initial_responses.append(response_text)

    if not any(initial_responses):
        return "Model did not provide any initial answers.", initial_responses

    synthesis_prompt = create_synthesis_prompt(
        original_question=f"Is there a connection between the attributes {', '.join(subgraph_nodes)}? "
                          f"Start the sentence with a Yes or No.",
        answers=initial_responses
    )

    # Model teraz bude generovať, kým neskončí alebo nenarazí na vysoký limit tokenov.
    final_output = llm(
        synthesis_prompt,
        max_tokens=2048,  # Ponechávame vysoký limit
        echo=False
    )
    final_answer = final_output['choices'][0]['text'].strip()

    return final_answer, initial_responses