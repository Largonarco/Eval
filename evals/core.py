import json
import asyncio
from bias import BiasEvaluator
from keys import OPEN_AI_API_KEY
from dataclasses import dataclass
from typing import Dict, List, Any
from accuracy import AccuracyEvaluator
from relevancy import AsyncRelevancyEvaluator


@dataclass
class CoreEvaluationResult:
    header: str
    overall_score: float
    strengths: List[str]
    coherence_score: float
    relevance_score: float
    interventions: List[str]
    bias_metrics: Dict[str, Any]
    content_scores: Dict[str, float]
    areas_for_improvement: List[str]
    specific_recommendations: List[str]

class CoreEvaluator:
    def __init__(self, openai_api_key: str, coherence_batch_size: int = 4):
        """Initialize evaluators with necessary configuration."""
        self.bias_evaluator = BiasEvaluator()
        self.accuracy_evaluator = AccuracyEvaluator(openai_api_key)
        self.relevancy_evaluator = AsyncRelevancyEvaluator(
            openai_api_key,
            coherence_batch_size=coherence_batch_size
        )

    async def evaluate_response(self, response: List[Dict], query: str = "") -> CoreEvaluationResult:
        """Evaluate a single response using all three evaluators."""
        
        # Get the header from the first block
        header = response[0].get('header', 'No Header')
        print(f"\n<------------------------------------------------------------>\n")
        print(f"HEADER: {header}\n")
        
        # Relevancy and Coherence evaluation
        metrics = await self.relevancy_evaluator.evaluate_response(query=query if query != "" else header, blocks=response)
        feedback = self.relevancy_evaluator.generate_actionable_feedback(metrics)

        print(f"RELEVANCY EVALUATION:\n")
        print(f"- Relevance Score: {metrics.relevance_score:.2f}")
        print(f"- Coherence Score: {metrics.coherence_score:.2f}")
        print(f"- Overall Score: {metrics.overall_score:.2f}")
        print("Feedback:")
        print("Strengths:")
        for strength in feedback['strengths']:
            print(f"- {strength}")
        print("Areas for Improvement:")
        for area in feedback['areas_for_improvement']:
            print(f"- {area}")
        print("Specific Recommendations:")
        for rec in feedback['specific_recommendations']:
            print(f"- {rec}")
        

        # Metric and Table Accuracy evaluation
        print("\nACCURACY EVALUATION:\n")
        for block in response:
            if 'metric' in block or 'table' in block:
                block_type = "metric" if block.get("metric", None) else "table"

                result = self.accuracy_evaluator.evaluate_block(block, block_type, header, {})
                feedback = self.accuracy_evaluator.generate_actionable_feedback(result)
                formatted_feedback = self.accuracy_evaluator.format_feedback(feedback)
                print(f"{formatted_feedback}")
								
                if result.content_score < 0.6:
                    # Get intervention suggestions
                    intervention = self.accuracy_evaluator.suggest_intervention(block, block_type, header, None, result)
                    print(f"{intervention}")
        
        # Bias evaluation
        print("GENDER BIAS EVALUATION:\n")
        bias_results = self.bias_evaluator.evaluate_bias(response)

        print("- Generic Representation Ratio: " + str(bias_results["bias_metrics"]["generic_representation_ratio"]))
        print("- Profession Association Ratio: " + str(bias_results["bias_metrics"]["professional_association_ratio"]))
        print("- Named Entities: " + str(int(bias_results["bias_metrics"]["named_entities"]["male_count"]) + int(bias_results["bias_metrics"]["named_entities"]["female_count"]))) 
        print("- Generic Mentions: " + str(bias_results["detailed_stats"]["generic_mentions"]))
        print("- Named Entity References: " + json.dumps(bias_results['detailed_stats']['named_entity_references'], indent=2))
        print("- Recommendations: " + str(bias_results["recommendations"]))


  
if __name__ == "__main__":
    evaluator = CoreEvaluator(openai_api_key=OPEN_AI_API_KEY, coherence_batch_size=5)  # Replace with your OpenAI API key

    with open('example_model_responses.json') as f:
        model_responses = json.load(f)
        
        for response in model_responses:
            asyncio.run(evaluator.evaluate_response(response))