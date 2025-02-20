import json
from openai import OpenAI
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class EvaluationResult:
    block_type: str
    has_citations: bool
    structure_valid: bool
    content_score: float
    error_message: str = None

class AccuracyEvaluator:
    def __init__(self, openai_api_key: str):
        """Initialize the evaluator with OpenAI API key."""
        self.client = OpenAI(api_key=openai_api_key)

    def _validate_structure(self, block: Dict, block_type: str) -> bool:
        """Validate the structure of a block based on its type."""
        if block_type == 'metric':
            return (
                isinstance(block.get('metric'), str) and
                isinstance(block.get('description'), str)
            )
        elif block_type == 'table':
            table_data = block.get('table', [])
            if not isinstance(table_data, list) or len(table_data) < 2:
                return False
                
            # Check if all rows have the same number of columns
            num_cols = len(table_data[0])
            
            # Check for consistent column count and no empty cells
            for row in table_data:
                # Check row length
                if len(row) != num_cols:
                    return False
                    
                # Check for empty cells
                for cell in row:
                    # Check for None, empty string, or whitespace-only string
                    if cell is None or (isinstance(cell, str) and not cell.strip()):
                        return False
                        
                    # If cell is any other type, ensure it has a non-empty string representation
                    if not str(cell).strip():
                        return False
            
            return True
            
        return False
    
    def _create_metric_evaluation_prompt(self, block: Dict, context: str, citation_content: Dict = None) -> str:
        """
        Create a prompt for evaluating a metric block with precise evaluation criteria.
        """
        citation_info = "No citations provided"
        if citation_content:
            citation_info = "Citation Content:\n"
            for cite_num, content in citation_content.items():
                citation_info += f"[{cite_num}]: {content}\n"
        
        return f"""
        Task: Evaluate the accuracy of the following metric in the given context.

        CONTENT TO EVALUATE:
        Context: {context}
        Metric: {block['metric']}
        Description: {block['description']}
        {citation_info}

        EVALUATION METHODOLOGY:
        1. Value Accuracy
           - Match exact metric value against evidence
           - Verify all significant digits
           - Check unit conversions if present

        2. Description Precision
           - Match description against source data
           - Verify temporal qualifiers (when/period)
           - Confirm statistical qualifiers (mean/median/mode)

        3. Contextual Alignment
           - Check data timeframe matches context
           - Verify geographic/demographic scope
           - Validate any comparisons or trends

        SCORING RUBRIC:
        0.81-1.00: Excellent
        - Value matches source exactly or within 1% margin
        - Description is complete and precise with all qualifiers
        - Strong citation support for all claims
        - Perfect contextual alignment
        
        0.61-0.80: Good
        - Value within 5% margin of error
        - Description accurate but missing minor qualifiers
        - Citations support most major claims
        - Minor contextual misalignments
        
        0.41-0.60: Acceptable
        - Value within 10-15% margin of error
        - Description lacks some important qualifiers
        - Partial citation support
        - Some contextual gaps
        
        0.21-0.40: Problematic
        - Value off by 15-25%
        - Description misrepresents key aspects
        - Weak or contradictory citation support
        - Major contextual issues
        
        0.00-0.20: Critical Issues
        - Value off by >25% or completely wrong
        - Description fundamentally inaccurate
        - Missing or irrelevant citations
        - Severe contextual misalignment

        INSTRUCTIONS:
        1. Follow methodology step by step
        2. Apply scoring rubric precisely
        3. Return single float between 0 and 1
        4. No explanation text

        RETURN ONLY THE NUMERICAL SCORE:"""

    def _create_table_evaluation_prompt(self, block: Dict, context: str, citation_content: Dict = None) -> str:
        """
        Create a prompt for evaluating a table block with precise evaluation criteria.
        """
        citation_info = "No citations provided"
        if citation_content:
            citation_info = "Citation Content:\n"
            for cite_num, content in citation_content.items():
                citation_info += f"[{cite_num}]: {content}\n"

        return f"""
        Task: Evaluate the accuracy of the following table in the given context.

        CONTENT TO EVALUATE:
        Context: {context}
        Table:
        {json.dumps(block['table'], indent=2)}
        {citation_info}

        EVALUATION METHODOLOGY:
        1. Cell Accuracy
           - Match each value against source
           - Verify units and conversions
           - Check significant digits

        2. Structural Correctness
           - Validate column/row headers
           - Verify data relationships
           - Check categorical groupings

        3. Completeness Check
           - Verify all required data present
           - Check for missing values
           - Validate data ranges

        SCORING RUBRIC:
        0.81-1.00: Excellent
        - 95%+ cells match source exactly or within 1% error
        - Headers perfectly labeled and organized
        - Complete data coverage with no gaps
        - All relationships and groupings correct
        
        0.61-0.80: Good
        - 85-94% cells within 5% error margin
        - Headers clear but could be more precise
        - Minor data gaps in non-critical areas
        - Most relationships and groupings accurate
        
        0.41-0.60: Acceptable
        - 70-84% cells within 10% error margin
        - Some header ambiguity present
        - Several data gaps but core data intact
        - Some grouping or relationship issues
        
        0.21-0.40: Problematic
        - 50-69% cells accurate within 15% error
        - Headers unclear or misleading
        - Significant data gaps affect understanding
        - Major grouping or relationship problems
        
        0.00-0.20: Critical Issues
        - <50% cells accurate or completely wrong
        - Headers missing or fundamentally wrong
        - Critical data missing throughout
        - Relationships and groupings invalid

        INSTRUCTIONS:
        1. Follow methodology step by step
        2. Apply scoring rubric precisely
        3. Return single float between 0 and 1
        4. No explanation text

        RETURN ONLY THE NUMERICAL SCORE:"""
    
    def _evaluate_content_accuracy(self, block: Dict, block_type: str, context: str, 
                                 citation_content: Dict = None) -> Tuple[float, Optional[Dict]]:
        """
        Evaluate the accuracy of block content using GPT-4.
        Returns a tuple of (accuracy_score, suggested_correction).
        """
        try:
            if block_type == 'metric':
                prompt = self._create_metric_evaluation_prompt(block, context, citation_content)
            else:  # table
                prompt = self._create_table_evaluation_prompt(block, context, citation_content)

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of factual accuracy in text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            # Extract score from response
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score)), None
            except ValueError:
                return 0.0, {"error": "Invalid score format received"}
            
        except Exception as e:
            return 0.0, {"error": str(e)}

    def evaluate_block(self, block: Dict, block_type: str, context: str = None, citations_data: Dict = {}) -> EvaluationResult:
        """
        Evaluate a single block (metric or table) for correctness.
        
        Args:
            block (Dict): The block to evaluate
            context (str, optional): Surrounding context for the block
            citations_data (Dict, optional): Dictionary mapping citation numbers to their content
                                          Format: {1: "citation text", 2: "citation text", ...}
        """ 
        # Basic citation checks
        block_citations = block.get('citations', [])
        has_citations = bool(block_citations)

        # Validate structure
        structure_valid = self._validate_structure(block, block_type)
        if not structure_valid:
            return EvaluationResult(
                block_type=block_type,
                has_citations=has_citations,
                structure_valid=False,
                content_score=0.0,
                error_message="Invalid block structure"
            )

        # Placeholder logic for citation content extraction as actual API is unknown
        citation_content = None
        if citations_data and block_citations:
            citation_content = {
                str(cite_num): citations_data.get(cite_num, None)
                for cite_num in block_citations
                if citations_data.get(cite_num, None) is not None
            }

        # Evaluate content accuracy using GPT
        content_score, meta = self._evaluate_content_accuracy(
            block, 
            block_type, 
            context, 
            citation_content
        )

        return EvaluationResult(
            block_type=block_type,
            has_citations=has_citations,
            structure_valid=structure_valid,
            content_score=content_score,
        )
    
    def generate_actionable_feedback(self, metrics: EvaluationResult) -> Dict:
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
            'specific_recommendations': [],
            'priority_level': 'low'  # Can be 'low', 'medium', 'high', 'critical'
        }
        
        # Structure validation feedback
        if not metrics.structure_valid:
            feedback['areas_for_improvement'].append('Invalid block structure detected')
            feedback['specific_recommendations'].extend([
                'Ensure all required fields are present',
                f'Verify the {metrics.block_type} follows the correct format specification',
                'Check for any missing or malformed data entries'
            ])
            feedback['priority_level'] = 'critical'

            return feedback

        # Citation feedback
        if not metrics.has_citations:
            feedback['areas_for_improvement'].append('Missing citations')
            feedback['specific_recommendations'].append('Add relevant citations to support the content')
            feedback['priority_level'] = 'high'

        # Content score-based feedback
        if metrics.content_score >= 0.81:  # Excellent (0.81-1.00)
            feedback['strengths'].extend([
                f'High-quality {metrics.block_type} content with excellent accuracy',
                'Precise and well-supported information',
                'Strong contextual alignment'
            ])

            if metrics.block_type == 'metric':
                feedback['strengths'].append('Accurate metric value with comprehensive description')
            else:  # table
                feedback['strengths'].append('Well-structured table with accurate cell values')
                
        elif metrics.content_score >= 0.61:  # Good (0.61-0.80)
            feedback['strengths'].append(f'Generally accurate {metrics.block_type} content')
            feedback['areas_for_improvement'].append('Minor accuracy improvements needed')

            if metrics.block_type == 'metric':
                feedback['specific_recommendations'].extend([
                    'Add missing qualifiers to the description',
                    'Verify exact metric value against source'
                ])
            else:
                feedback['specific_recommendations'].extend([
                    'Review header clarity and precision',
                    'Check for minor data inconsistencies'
                ])
            feedback['priority_level'] = 'low'
                
        elif metrics.content_score >= 0.41:  # Acceptable (0.41-0.60)
            feedback['areas_for_improvement'].extend([
                'Moderate accuracy issues detected',
                'Important details missing or imprecise'
            ])

            if metrics.block_type == 'metric':
                feedback['specific_recommendations'].extend([
                    'Review and verify metric calculation',
                    'Add important contextual qualifiers',
                    'Strengthen citation support'
                ])
            else:
                feedback['specific_recommendations'].extend([
                    'Address data gaps in the table',
                    'Improve header clarity',
                    'Verify data relationships and groupings'
                ])
            feedback['priority_level'] = 'medium'
                
        elif metrics.content_score >= 0.21:  # Problematic (0.21-0.40)
            feedback['areas_for_improvement'].extend([
                'Significant accuracy concerns',
                'Major content issues identified'
            ])

            if metrics.block_type == 'metric':
                feedback['specific_recommendations'].extend([
                    'Recalculate metric value from source data',
                    'Completely revise description for accuracy',
                    'Verify all citations and claims'
                ])
            else:
                feedback['specific_recommendations'].extend([
                    'Review and correct cell values throughout',
                    'Restructure table headers and organization',
                    'Fill in missing critical data'
                ])
            feedback['priority_level'] = 'high'
                
        else:  # Critical Issues (0.00-0.20)
            feedback['areas_for_improvement'].extend([
                'Critical accuracy issues detected',
                'Fundamental content problems present'
            ])

            if metrics.block_type == 'metric':
                feedback['specific_recommendations'].extend([
                    'Complete metric value recalculation required',
                    'Full revision of description needed',
                    'Comprehensive citation review necessary'
                ])
            else:
                feedback['specific_recommendations'].extend([
                    'Complete table restructuring needed',
                    'Comprehensive data verification required',
                    'Full review of all relationships and groupings'
                ])
            feedback['priority_level'] = 'critical'

        return feedback

    def format_feedback(self, feedback: Dict) -> str:
        """
        Format the feedback dictionary into a readable string.
        
        Parameters:
        - feedback: Dictionary containing feedback sections
        
        Returns:
        - Formatted string with feedback
        """
        priority_indicators = {
            'low': '🟢',
            'high': '🟠',
            'medium': '🟡',
            'critical': '🔴'
        }
        formatted = f"Priority Level: {priority_indicators[feedback['priority_level']]} {feedback['priority_level']}\n"
        
        if feedback['strengths']:
            formatted += "✓ Strengths:\n"
            formatted += "\n".join(f"  • {strength}" for strength in feedback['strengths'])
            formatted += "\n"
            
        if feedback['areas_for_improvement']:
            formatted += "⚠ Areas for Improvement:\n"
            formatted += "\n".join(f"  • {area}" for area in feedback['areas_for_improvement'])
            formatted += "\n"
            
        if feedback['specific_recommendations']:
            formatted += "📋 Specific Recommendations:\n"
            formatted += "\n".join(f"  • {rec}" for rec in feedback['specific_recommendations'])
            formatted += "\n"
            
        return formatted
    
    # Intervening and Correction Logic
    def _create_correction_prompt(self, block: Dict, block_type: str, context: str, 
                                citation_content: Dict, evaluation_result: EvaluationResult) -> str:
        """
        Create a prompt for correcting inaccurate content based on evaluation results.
        
        Args:
            block: The original block content
            block_type: Type of block ('metric' or 'table')
            context: Surrounding context
            citation_content: Available citation content
            evaluation_result: Previous evaluation results
        """
        has_citations = citation_content and len(citation_content) > 0

        citation_info = "No citations provided"
        if citation_content:
            citation_info = "Citation Content:\n"
            for cite_num, content in citation_content.items():
                citation_info += f"[{cite_num}]: {content}\n"

        if block_type == 'metric':
            return f"""
            Task: Correct the following metric content to improve accuracy.

            ORIGINAL CONTENT:
            Context: {context}
            Original Metric: {block['metric']}
            Original Description: {block['description']}
            {citation_info}

            CORRECTION GUIDELINES:
            1. {"Maintain exact alignment with citation content" if has_citations else "Suggest corrections based on the context and your expertise"}
            2. Include all necessary qualifiers (temporal, statistical, geographic)
            3. Use precise language and specific values
            4. Keep the same basic format but improve accuracy
            {"5. Preserve the original citation structure" if has_citations else ""}

            REQUIRED OUTPUT FORMAT:
            Return a JSON object with corrected 'metric' and 'description' fields:
            {{
                "metric": "corrected metric value",
                "description": "corrected description",
            }}
            ONLY RETURN JSON OBJECT 
            Generate the corrected content:"""
        else:  # table
            return f"""
            Task: Correct the following table content to improve accuracy.

            ORIGINAL CONTENT:
            Context: {context}
            Original Table:
            {json.dumps(block['table'], indent=2)}
            {citation_info}

            CORRECTION GUIDELINES:
            1. {"Maintain exact alignment with citation content" if has_citations else "Suggest corrections based on the context and your expertise"}
            2. Preserve table structure and relationships
            3. Correct any numerical inaccuracies
            4. Ensure header clarity and precision
            5. Keep the same format but improve accuracy

            REQUIRED OUTPUT FORMAT:
            Return a JSON object with the corrected table data:
            {{
                "table": [corrected table rows],
            }}
            ONLY RETURN JSON OBJECT WITH CORRECTED TABLE DATA
            Generate the corrected content:"""

    def _generate_correction(self, block: Dict, block_type: str, context: str,
                           citation_content: Dict, evaluation_result: EvaluationResult) -> Dict:
        """
        Generate corrected content for a block that failed accuracy checks.
        """
        try:
            correction_prompt = self._create_correction_prompt(
                block, block_type, context, citation_content, evaluation_result
            )

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at correcting factual inaccuracies in content while maintaining citation integrity."},
                    {"role": "user", "content": correction_prompt}
                ],
                temperature=0.1
            )

            corrected_content = json.loads(response.choices[0].message.content.strip())
            
            # Validate the correction format
            if block_type == 'metric':
                if not all(key in corrected_content for key in ['metric', 'description']):
                    raise ValueError("Invalid correction format for metric")
            else:  # table
                if not all(key in corrected_content for key in ['table']):
                    raise ValueError("Invalid correction format for table")
                
            return corrected_content
        except Exception as e:
            return {
                "error": f"Failed to generate correction: {str(e)}",
                "original_content": block
            }

    def suggest_intervention(self, block: Dict, block_type: str, context: str,
                           citation_content: Dict, evaluation_result: EvaluationResult) -> Dict:
        """
        Suggest intervention strategies based on evaluation results and generate corrections when needed.
        
        Args:
            block: The original block content
            block_type: Type of block ('metric' or 'table')
            context: Surrounding context
            citation_content: Available citation content
            evaluation_result: Previous evaluation results
            
        Returns:
            Dictionary with intervention recommendations and corrections if needed
        """
        formatted_intervention = "Suggested correction:\n"

        # Generate corrections based on accuracy score
        corrected_content = self._generate_correction(
                block, block_type, context, citation_content, evaluation_result
            )
    
        # Handle the corrected content based on its type
        if isinstance(corrected_content, dict):
            if "error" in corrected_content:
                # Handle error case
                formatted_intervention += f"Error generating correction: {corrected_content['error']}\n"
            else:
                # Format the dictionary content
                formatted_intervention += json.dumps(corrected_content, indent=2)
                formatted_intervention += "\n"
        else:
            # If corrected_content is already a string, add it directly
            formatted_intervention += str(corrected_content)
            formatted_intervention += "\n"

        return formatted_intervention

def main():
    evaluator = AccuracyEvaluator("xxxxx")  # Replace 'xxxxx' with your OpenAI API key
    
	# Load example_model_responses.json file and loop over all outputs and their respective blocks	
    with open('example_model_responses.json') as f:
        model_responses = json.load(f)
        
        for response in model_responses:
          header = response[0].get('header', 'No Header')
          print(f"\n<------------------------------------------------------------>\n")
          print(f"HEADER: {header}\n")
          
          for block in response:
              if 'header' in block:
                  header = block['header']

              if 'metric' in block or "table" in block:
                block_type = "metric" if block.get("metric", None) else "table"

				# Evaluate the block
                result = evaluator.evaluate_block(block, block_type, header, {})
                feedback = evaluator.generate_actionable_feedback(result)
                formatted_feedback = evaluator.format_feedback(feedback)
                print(f"{formatted_feedback}")
								
                if result.content_score < 0.6:
                    # Get intervention suggestions
                    intervention = evaluator.suggest_intervention(block, block_type, header, None, result)
                    print(f"{intervention}")
	              
               

if __name__ == "__main__":
    main()