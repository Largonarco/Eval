import json
import asyncio
import numpy as np
from openai import AsyncOpenAI
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class EvaluationMetrics:
    relevance_score: float
    coherence_score: float
    overall_score: float

class AsyncRelevancyEvaluator:
    def __init__(self, openai_api_key: str, coherence_batch_size: int = 4):
        """Initialize the evaluator with OpenAI API key and batch size for parallel processing."""
        self.coherence_batch_size = coherence_batch_size
        self.client = AsyncOpenAI(api_key=openai_api_key)

    def _create_coherence_prompt(self, current_block: Dict, next_block: Dict) -> str:
        """
        Helper method to create the coherence evaluation prompt.
        """
        return f"""
            Task: Evaluate the coherence and logical flow between two consecutive content blocks.

            CONTENT TO EVALUATE:
            Block 1: {current_block.get('paragraph', current_block.get('header', ''))}
            Block 2: {next_block.get('paragraph', next_block.get('header', ''))}

            EVALUATION METHODOLOGY:
            1. Logical Flow Assessment:
              - Evaluate the logical connection between blocks
              - Check for proper sequence of ideas
              - Assess if transitions are natural

            2. Content Continuity:
              - Verify thematic consistency
              - Check for information gaps
              - Assess proper development of ideas

            3. Structural Coherence:
              - Evaluate transition effectiveness
              - Check for appropriate segmentation
              - Assess paragraph-level organization

            SCORING RUBRIC:
            0.0-0.2: No coherence
            - No logical connection between blocks
            - Abrupt or jarring transitions
            - Completely disconnected ideas

            0.3-0.4: Weak coherence
            - Minimal logical connection
            - Poor transitions
            - Significant gaps in flow

            0.5-0.6: Moderate coherence
            - Basic logical connection
            - Functional transitions
            - Some gaps in flow

            0.7-0.8: Strong coherence
            - Clear logical connection
            - Smooth transitions
            - Minor gaps in flow

            0.9-1.0: Excellent coherence
            - Perfect logical connection
            - Seamless transitions
            - Flawless flow of ideas

            INSTRUCTIONS:
            1. Follow the evaluation methodology step by step
            2. Apply the scoring rubric rigorously
            3. Return ONLY a single float number between 0 and 1 representing the coherence score
            4. Do not provide any explanation or additional text

            RETURN ONLY THE NUMERICAL SCORE:
            """
    
    def _create_relevance_prompt(self, query: str, full_response: str) -> str:
        """
        Helper method to create the relevance evaluation prompt.
        """
        return f"""
        Task: Evaluate the relevance and accuracy of the following complete response in relation to the query.

        CONTENT TO EVALUATE:
        Query: {query}
        Complete Response: {full_response}

        EVALUATION METHODOLOGY:
        1. Query Alignment Assessment:
          - Evaluate how comprehensively the response addresses the query
          - Check if all aspects of the query are covered
          - Assess the depth and thoroughness of the response

        2. Content Quality Assessment:
          - Verify accuracy and completeness of information
          - Check for logical consistency throughout the response
          - Evaluate clarity and effectiveness of explanations

        3. Overall Response Effectiveness:
          - Assess if the response fully satisfies the query intent
          - Evaluate the balance of information provided
          - Check for any missing crucial information

        SCORING RUBRIC:
        0.0-0.2: Irrelevant or misleading
        - Response fails to address the query
        - Contains significant inaccuracies
        - Missing crucial information

        0.3-0.4: Minimally relevant
        - Superficially addresses the query
        - Contains notable gaps or inaccuracies
        - Lacks necessary depth

        0.5-0.6: Moderately relevant
        - Partially addresses the query
        - Generally accurate with some gaps
        - Provides basic but incomplete coverage

        0.7-0.8: Highly relevant
        - Comprehensively addresses the query
        - Accurate with minor gaps
        - Provides thorough coverage

        0.9-1.0: Exceptionally relevant
        - Perfectly addresses all aspects of the query
        - Completely accurate and comprehensive
        - Provides exceptional depth and clarity

        INSTRUCTIONS:
        1. Follow the evaluation methodology step by step
        2. Apply the scoring rubric rigorously
        3. Return ONLY a single float number between 0 and 1 representing the relevance score
        4. Do not provide any explanation or additional text

        RETURN ONLY THE NUMERICAL SCORE:
        """
    
    async def _evaluate_block_pair_coherence(self, current_block: Dict, next_block: Dict) -> float:
        """
        Helper method to evaluate coherence between two consecutive blocks.
        """
        prompt = self._create_coherence_prompt(current_block, next_block)
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            score_text = response.choices[0].message.content.strip()

            return float(score_text)
        except (ValueError, AttributeError):
            return 0.5

    async def evaluate_coherence(self, blocks: List[Dict]) -> float:
        """
        Evaluate coherence between all blocks in batches.
        This function handles the complete coherence evaluation pipeline including:
        - Creating block pairs
        - Processing batches in parallel
        - Aggregating results
        """
        # Create pairs of consecutive blocks for coherence evaluation
        block_pairs = list(zip(blocks[:-1], blocks[1:]))
        
        # If no pairs to evaluate, return default score
        if not block_pairs:
            return 0.5

        # Process all pairs in batches
        all_scores = []
        for i in range(0, len(block_pairs), self.coherence_batch_size):
            batch = block_pairs[i:i + self.coherence_batch_size]
            # Create tasks for each pair in the current batch
            batch_tasks = [
                self._evaluate_block_pair_coherence(current_block, next_block)
                for current_block, next_block in batch
            ]
            # Run batch tasks concurrently
            batch_scores = await asyncio.gather(*batch_tasks)
            all_scores.extend(batch_scores)

        # Calculate and return mean coherence score
        return np.mean(all_scores)

    async def evaluate_relevance(self, blocks: List[Dict], query: str) -> float:
        """
        Evaluate the overall relevance of the response to the query.
        This function handles the complete relevance evaluation including:
        - Combining blocks into full response
        - Evaluating overall relevance
        - Error handling and fallback
        """
        # Combine all blocks into a single coherent text
        full_response = "\n\n".join([
            block.get('paragraph', block.get('header', '')) 
            for block in blocks
        ])
        
        prompt = self._create_relevance_prompt(query, full_response)
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            score_text = response.choices[0].message.content.strip()

            return float(score_text)
        except (ValueError, AttributeError):
            return 0.5

    async def evaluate_response(self, query: str, blocks: List[Dict]) -> EvaluationMetrics:
        """
        Main evaluation pipeline that coordinates parallel processing of 
        coherence and relevance evaluations.
        """
        # Start both evaluations concurrently
        coherence_task = asyncio.create_task(self.evaluate_coherence(blocks))
        relevance_task = asyncio.create_task(self.evaluate_relevance(blocks, query))
        
        # Wait for both tasks to complete
        coherence_score, relevance_score = await asyncio.gather(coherence_task, relevance_task)

        # Calculate weighted overall score
        weights = {'relevance': 0.6, 'coherence': 0.4}
        overall_score = (
            weights['relevance'] * relevance_score +
            weights['coherence'] * coherence_score 
        )

        return EvaluationMetrics(
            relevance_score=relevance_score,
            coherence_score=coherence_score,
            overall_score=overall_score
        )
    
    def generate_actionable_feedback(self, metrics: EvaluationMetrics) -> Dict:
        """
        Generate specific, actionable feedback based on evaluation metrics.
        
        Parameters:
        - metrics: EvaluationMetrics object containing evaluation scores
        
        Returns:
        - Dictionary containing actionable feedback and improvement suggestions
        """
        feedback = {
            'strengths': [],
            'areas_for_improvement': [],
            'specific_recommendations': []
        }
        
        # Generate specific feedback for overall response relevance
        if metrics.relevance_score < 0.8:
            feedback['areas_for_improvement'].append(
                'Overall response relevance needs improvement'
            )
            feedback['specific_recommendations'].append(
                'Ensure the response comprehensively addresses all aspects of the query'
            )
            
        if metrics.coherence_score < 0.8:
            feedback['areas_for_improvement'].append(
                'Response coherence could be strengthened'
            )
            feedback['specific_recommendations'].append(
                'Improve logical flow and transitions throughout the response'
            )
            
        # Identify strengths
        if metrics.relevance_score >= 0.8:
            feedback['strengths'].append(
                'Excellent overall relevance to query'
            )
            
        if metrics.coherence_score >= 0.8:
            feedback['strengths'].append(
                'Strong coherence and logical flow'
            )
            
        return feedback


    
async def main():
    evaluator = AsyncRelevancyEvaluator(
        "xxxxxx",  # Add your OpenAI API key here
        coherence_batch_size=4
    )
    
    # Load and evaluate responses
    with open('example_model_responses.json') as f:
        model_responses = json.load(f)
        
        for response in model_responses:
            header = response[0].get('header', 'No Header')
            print(f"\n<------------------------------ {header} ------------------------------>\n")

            metrics = await evaluator.evaluate_response(query="How does quantum superposition differ from classical binary states?", blocks=response)
            feedback = evaluator.generate_actionable_feedback(metrics)
            
            print(f"Evaluation Metrics:")
            print(f"- Relevance Score: {metrics.relevance_score:.2f}")
            print(f"- Coherence Score: {metrics.coherence_score:.2f}")
            print(f"- Overall Score: {metrics.overall_score:.2f}\n")
            
            print("Feedback:")
            print("Strengths:")
            for strength in feedback['strengths']:
                print(f"- {strength}")
            
            print("\nAreas for Improvement:")
            for area in feedback['areas_for_improvement']:
                print(f"- {area}")
            
            print("\nSpecific Recommendations:")
            for rec in feedback['specific_recommendations']:
                print(f"- {rec}")

if __name__ == "__main__":
    asyncio.run(main())