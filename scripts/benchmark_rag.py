"""
Benchmark Script: Standard RAG vs Graph-Expanded RAG
Evaluates retrieval quality by comparing answers using an LLM as a judge.
"""
import sys
import os
import json

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.retrieval.search import HybridSearcher
from backend.services import LLMService

TEST_QUERIES = [
    "Where is sumState used?",
    "How is raw sales data loaded into ClickHouse?",
    "What partitions are used in sales_aggregates?",
    "Explain the analytics config structure",
    "What tables does the ETL pipeline write to?"
]

def evaluate_answers(query, ans_standard, ans_graph):
    """Uses LLM to blindly judge which answer is better."""
    system_prompt = (
        "You are an impartial but strict technical judge evaluating two RAG (Retrieval-Augmented Generation) systems.\n"
        "You will be given a query and two answers (Answer A and Answer B).\n"
        "Evaluate them strictly based on completeness, accuracy, and technical detail.\n"
        "Use this strict rubric:\n"
        "- 1-4: Incorrect, irrelevant, or highly incomplete.\n"
        "- 5-6: Bare minimum answer. Technically correct but lacks depth or context.\n"
        "- 7: Good answer, but noticeably missing some broader architectural context or file citations.\n"
        "- 8: Very good answer. Accurate and complete, which should be the standard for a solid retrieval.\n"
        "- 9: Excellent answer. Brings in adjacent context, configuration details, or specific file names.\n"
        "- 10: Flawless. Provides exhaustive context and leaves nothing out.\n"
        "Output ONLY a JSON object with your evaluation, using this exact format:\n"
        "{\"reasoning\": \"Detailed explanation of your evaluation\", \"score_a\": 8, \"score_b\": 9}"
    )
    
    user_prompt = f"Query: {query}\n\n--- Answer A (Standard RAG) ---\n{ans_standard}\n\n--- Answer B (Graph RAG) ---\n{ans_graph}"
    
    try:
        response = LLMService.chat_completion(system_prompt, user_prompt)
        # Clean markdown codeblocks if LLM included them
        if response.startswith("```json"):
            response = response[7:-3]
        return json.loads(response.strip())
    except Exception as e:
        print(f"Error parsing LLM evaluation: {e}")
        return {"score_a": 0, "score_b": 0, "reasoning": "Failed to evaluate"}

def run_benchmarks():
    searcher = HybridSearcher()
    
    results = []
    
    print("Running Graph vs Standard RAG Benchmarks...")
    
    for i, query in enumerate(TEST_QUERIES):
        print(f"\n[Query {i+1}/{len(TEST_QUERIES)}] {query}")
        
        # 1. Run Standard RAG (no graph)
        print("  -> Running Standard RAG...")
        res_standard = searcher.perform_rag_query(query, use_graph=False)
        ans_standard = res_standard["answer"]
        
        # 2. Run Graph RAG
        print("  -> Running Graph-Expanded RAG...")
        res_graph = searcher.perform_rag_query(query, use_graph=True)
        ans_graph = res_graph["answer"]
        
        # 3. Evaluate
        print("  -> LLM Judging...")
        eval_res = evaluate_answers(query, ans_standard, ans_graph)
        print(f"     Standard: {eval_res['score_a']}/10 | Graph: {eval_res['score_b']}/10")
        
        results.append({
            "query": query,
            "standard_score": eval_res["score_a"],
            "graph_score": eval_res["score_b"],
            "reasoning": eval_res["reasoning"]
        })
        
    print("\n\n" + "="*80)
    print("BENCHMARK RESULTS (MARKDOWN TABLE)")
    print("="*80 + "\n")
    
    markdown_lines = []
    markdown_lines.append("| Query | Standard RAG | Graph-Expanded RAG | Win | Reasoning |")
    markdown_lines.append("|-------|--------------|--------------------|-----|-----------|")
    
    wins = {"Standard": 0, "Graph": 0, "Tie": 0}
    
    for r in results:
        diff = r["graph_score"] - r["standard_score"]
        if diff > 0:
            win = "Graph"
            wins["Graph"] += 1
        elif diff < 0:
            win = "Standard"
            wins["Standard"] += 1
        else:
            win = "Tie"
            wins["Tie"] += 1
            
        markdown_lines.append(f"| {r['query']} | {r['standard_score']}/10 | {r['graph_score']}/10 | {win} | {r['reasoning']} |")
        
    markdown_lines.append(f"\n**Final Score:** Graph {wins['Graph']} - Standard {wins['Standard']} - Ties {wins['Tie']}")
    
    output_path = r"C:\Users\Diacto\.gemini\antigravity-ide\brain\0c6c321c-865d-4242-987f-580342f7d15a\benchmark_results.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))
    print(f"Successfully wrote results to {output_path}")

if __name__ == "__main__":
    run_benchmarks()
