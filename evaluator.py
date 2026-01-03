import time
import json
from typing import List, Dict, Any, Tuple
from langchain_ollama import OllamaLLM

# ============================================================================
# CONFIGURATION
# ============================================================================
LLM_NAME = "gemma3:27b"

KNOWN_AIRLINES = ['lufthansa', 'britishairways', 'airindia', 'singaporeairlines']

# ============================================================================
# LLM-AS-A-JUDGE EVALUATOR CLASS
# ============================================================================
class LLMJudgeEvaluator:
    """LLM-as-a-Judge evaluation system with hybrid scoring."""

    def __init__(self, judge_model: str = LLM_NAME):
        self.judge_llm = OllamaLLM(model=judge_model, temperature=0.0)
        self.standard_rag = None
        self.graph_rag = None

    def set_rag_systems(self, standard_rag, graph_rag):
        """Set the RAG systems for evaluation."""
        self.standard_rag = standard_rag
        self.graph_rag = graph_rag

    def create_judge_prompt(self, question: str, reference: str, prediction: str) -> str:
        """Create evaluation prompt for the judge LLM with focus on content over formatting."""
        return f"""You are an expert evaluator assessing RAG system responses for multi-hop booking policy questions. Your task is to evaluate whether the prediction correctly answers the question based on factual accuracy and completeness.

    IMPORTANT INSTRUCTIONS:
    - IGNORE all section headers, prefixes, or airline codes (e.g., "BA", "AI", "SIA", "LH", "British Airways", "airindia", "singaporeairlines", "lufthansa", etc.) in both reference and prediction
    - Do NOT penalize the prediction for including such references IDs or labels as long as the factual content and conditions are correct
    - Focus ONLY on the factual content and policy conditions stated in the answers
    - Evaluate semantic equivalence, not literal text matching
    - Different wording is acceptable if the meaning and all conditions are preserved

    EVALUATION CRITERIA (in order of importance):

    1. COMPLETENESS (40 points): Does the prediction include ALL key facts, conditions, and requirements from the reference?
       - Check each policy condition, restriction, or requirement is present
       - Verify all entities (fees, timeframes, exceptions) are mentioned
       - For multi-hop questions, ensure all reasoning steps are covered

    2. FACTUAL ACCURACY (40 points): Are all facts, numbers, and conditions in the prediction correct and consistent with the reference?
       - Verify fees, percentages, time periods match exactly
       - Check policy conditions are not contradicted or altered
       - Ensure no hallucinated information is added

    3. RELEVANCE (10 points): Does the prediction directly answer the question asked?
       - Answer addresses the specific query
       - No off-topic or unnecessary information

    4. CLARITY (10 points): Is the response well-structured and understandable?
       - Logical organization of information
       - Clear communication of policy details

    SCORING SCALE: 0-100
    - 90-100: All conditions met, semantically equivalent to reference
    - 70-89: Minor omissions or slight inaccuracies that don't affect core answer
    - 50-69: Missing significant conditions or contains factual errors
    - 30-49: Partially correct but major gaps in completeness or accuracy
    - 0-29: Mostly incorrect, irrelevant, or missing critical information

    QUESTION: {question}

    REFERENCE ANSWER (ground truth): {reference}

    PREDICTED ANSWER (to evaluate): {prediction}

    EVALUATION PROCESS:
    1. Extract key facts and conditions from reference (ignoring headers/formatting)
    2. Check if prediction contains each fact/condition (ignoring exact wording)
    3. Identify any missing information or factual errors
    4. Calculate score based on completeness and accuracy

    Provide your evaluation as valid JSON with exactly these keys:
    {{"score": <number 0-100>, "reason": "<explanation covering: (a) key facts present, (b) missing information if any, (c) factual errors if any, (d) overall assessment>"}}"""

    def parse_judge_response(self, response_text: str) -> Tuple[int, str]:
        """Parse judge LLM response to extract score and reason."""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)

                if "score" in result and "reason" in result:
                    score = int(result["score"])
                    score = max(0, min(100, score))
                    return score, result["reason"]

            return 0, "Failed to parse judge response"
        except Exception as e:
            return 0, f"Error parsing response: {str(e)}"

    def parse_section_ids(self, section_ids_str: str) -> List[str]:
        """Parse section IDs from CSV string to list.

        Handles formats:
        - "BA11,BA12,BA32"
        - "BA11, BA12, BA32"
        - "[BA11, BA12, BA32]"
        """
        if not section_ids_str or section_ids_str.strip() == "":
            return []


        # Split by comma and clean
        section_ids = [sid.strip().strip('"').strip("'") for sid in section_ids_str.split(',')]

        # Filter out empty strings
        return [sid for sid in section_ids if sid]

    def calculate_section_match_score(self, retrieved_section_ids: List[str],
                                      correct_section_ids: List[str]) -> Tuple[float, str]:
        """Calculate section matching score.

        Args:
            retrieved_section_ids: Sections retrieved by the system
            correct_section_ids: Ground truth correct sections

        Returns:
            (score 0-100, explanation)
        """
        if not correct_section_ids:
            return 100.0, "No ground truth sections specified"

        if not retrieved_section_ids:
            return 0.0, "No sections retrieved by system"

        # Convert to sets
        correct_set = set(correct_section_ids)
        retrieved_set = set(retrieved_section_ids)

        # Calculate matches
        matched_sections = correct_set.intersection(retrieved_set)
        num_matched = len(matched_sections)
        num_correct = len(correct_set)

        # Score = (matched / total_correct) * 100
        score = (num_matched / num_correct) * 100.0

        # Create explanation
        explanation = f"Matched {num_matched}/{num_correct} correct sections. "
        if matched_sections:
            explanation += f"Found: {sorted(list(matched_sections))}. "

        return score, explanation.strip()

    async def evaluate_single_query(self, question: str, reference: str, system_type: str,
                                    answer_section_ids: str = None) -> Dict[str, Any]:
        """Evaluate query with HYBRID SCORING: 50% LLM + 50% section matching.

        Args:
            question: Query question
            reference: Expected answer
            system_type: "Standard RAG" or "GraphRAG"
            answer_section_ids: Comma-separated correct section IDs (optional)
        """
        start_time = time.time()

        try:
            # Get prediction from RAG system
            if system_type == "Standard RAG":
                if not self.standard_rag or not self.standard_rag.initialized:
                    return {"error": "Standard RAG not initialized", "score": 0, "question": question,
                            "reference": reference}
                result = await self.standard_rag.query_async(question)
            elif system_type == "GraphRAG":
                if not self.graph_rag or not self.graph_rag.initialized:
                    return {"error": "GraphRAG not initialized", "score": 0, "question": question,
                            "reference": reference}
                result = await self.graph_rag.query_async(question)
            else:
                return {"error": "Invalid system type", "score": 0, "question": question, "reference": reference}

            if not result.get("success"):
                return {
                    "question": question,
                    "reference": reference,
                    "error": result.get("error", "Unknown error"),
                    "score": 0,
                    "llm_score": 0,
                    "section_score": 0,
                    "prediction": "Error generating answer",
                    "response_time": time.time() - start_time,
                    "reason": "System error",
                    "section_explanation": "N/A"
                }

            prediction = result["answer"]

            # PART 1: LLM Judge (50%)
            judge_prompt = self.create_judge_prompt(question, reference, prediction)
            judge_response = self.judge_llm.invoke(judge_prompt)
            llm_score, llm_reason = self.parse_judge_response(judge_response)

            # PART 2: Section Matching (50%)
            section_score = 100.0
            section_explanation = "Section matching N/A (no ground truth)"

            if answer_section_ids:
                correct_section_ids = self.parse_section_ids(answer_section_ids)
                retrieved_section_ids = result.get("section_ids", [])
                section_score, section_explanation = self.calculate_section_match_score(
                    retrieved_section_ids, correct_section_ids
                )

            # HYBRID SCORE: 50% + 50%
            final_score = (llm_score * 0.5) + (section_score * 0.5)

            return {
                "question": question,
                "system_type": system_type,
                "reference_answer": reference,
                "predicted_answer": prediction,
                "total_score": round(final_score, 1),
                "llm_score": llm_score,
                "section_score": round(section_score, 1),
                "llm_score_explanation": llm_reason,
                "section_score_explanation": section_explanation,
                "response_time": time.time() - start_time,
                "retrieved_sections": result.get("section_ids", [])
            }

        except Exception as e:
            return {
                "question": question,
                "reference_answer": reference,
                "error": str(e),
                "total_score": 0,
                "llm_score": 0,
                "section_score": 0,
                "predicted_answer": f"Error: {str(e)}",
                "response_time": time.time() - start_time,
                "llm_score_explanation": "Exception occurred",
                "section_score_explanation": "N/A"
            }

