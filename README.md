# The Task

The goal is to design evaluations for our system's outputs. Our system ingests queries and data to produce a cited and multi-media response. LLMS are at the core of our system which poses challenges with evaluating performance and making informed design choices. Manual inspection of results is generally not feasible due to the time and expertise involved. Hence the need for a robust, automated evaluation system.

Our system's response is internally represented by JSON objects that we call "blocks". It is the array of blocks representing the final document, or the individual blocks, that you will be evaluating. The blocks we currently support are:

1. Paragraph
2. Header
3. AI Image
4. Web Image
5. AI Chart
6. Web Chart
7. Quote
8. Metric
9. Table
10. Tweet

You can see examples of full system outputs and of all these blocks in the `example_model_responses.json` file. However, note that these examples were produced some months ago.

Task specification:

1. Write an evaluation for bias in the output. This could be implicit or explicit bias. You may narrow the evaluation to a particular type of bias, such as gender or political bias. Extra credit: Write a script to probe if our system can be steered towards biased outputs, using your evaluation as a component.
2. Write an evaluation for identifying if metric and table blocks contain correct data. Extra credit: Use your evaluation to create an intervention strategy that identifies, and possibly corrects for, incorrect metrics or tables. Extra extra credit: Make the intervention strategy low-latency.
3. Write an evaluation for a problem of your choice that you think is important to address.
4. Write down the thoughts behind your approach in the `approach.md` file. Be detailed. Include your findings, limitations of your results, and future directions.

Consider if the results of the evaluation are actionable. A common issue with submissions is that the results are not informative or precise enough to be used for real decision making. If there is anything you don't have time to do, spend time thinking about what you would do with more time, since I may ask about this.

# Getting Started

### 1. Validate Your OpenAI Key

```
python3 validate_openai_key.py
True
```

### 2. Run a Small Test

```
python3 api_utils.py
[
    {
        "header": "Introduction to String Theory"
    },
    {
        "paragraph": "String theory is a theoretical framework in physics that seeks to unify the principles of General Relativity and Quantum Mechanics. Unlike traditional particle physics, which describes particles as point-like entities, string theory posits that the fundamental components of matter are one-dimensional strings that vibrate at different frequencies. These vibrations correspond to various particles, such as quarks and electrons, fundamentally altering our understanding of the universe's building blocks."
    },
    {
        "paragraph": "The vibrational states of strings determine the properties of particles, including their mass and charge. Strings can be open or closed, and their interactions introduce a degree of non-locality, allowing them to interact at any point along their length. This contrasts sharply with the Standard Model of particle physics, which relies on point-like particles and does not incorporate gravity."
    },
    {
        "table": {
            "data": [
                [
                    "Property",
                    "String Theory",
                    "Standard Model"
                ],
                [
                    "Fundamental Components",
                    "One-dimensional strings",
                    "Point-like particles"
                ],
                [
                    "Inclusion of Gravity",
                    "Yes",
                    "No"
                ],
                [
                    "Dimensions Required",
                    "10 or 11",
                    "4 (3 spatial + 1 time)"
                ]
            ]
        }
    },
    {
        "number": "10 or 11 dimensions",
        "description": "String theory typically requires the existence of 10 or 11 dimensions beyond our observable universe."
    },
    {
        "paragraph": "The implications of string theory extend beyond particle physics, suggesting a more complex structure of the universe. It opens up possibilities for parallel universes and alternative models of cosmology, such as the ekpyrotic universe, where our universe results from the collision of branes. This challenges our traditional notions of space, time, and the fundamental nature of reality."
    }
]
```

The script `generate_model_responses.py` will create documents in parallel by sampling queries from the `example_queries.json` file. The `example_model_responses.json` shows outputs (from a few months ago) using mostly the default payload (except the model is claude) on one randomly sampled query from each category. You can use these example responses to save time spent generating outputs. Although there are some things you will not be able to test using the example responses because they were all generated from the stock payload.

### 3. Run Evals

Before running the evaluation pipeline add the OpenAI key in evals/core.py main function.

```
python evals/core.py
<------------------------------------------------------------>

HEADER: Quantum Superposition vs. Classical Binary States: Unveiling the Quantum Advantage

RELEVANCY EVALUATION:

- Relevance Score: 1.00
- Coherence Score: 0.34
- Overall Score: 0.74
Feedback:
Strengths:
- Excellent overall relevance to query
Areas for Improvement:
- Response coherence could be strengthened
Specific Recommendations:
- Improve logical flow and transitions throughout the response

ACCURACY EVALUATION:

Priority Level: 🟢 low
✓ Strengths:
  • High-quality metric content with excellent accuracy
  • Precise and well-supported information
  • Strong contextual alignment
  • Accurate metric value with comprehensive description

Priority Level: 🟢 low
✓ Strengths:
  • High-quality table content with excellent accuracy
  • Precise and well-supported information
  • Strong contextual alignment
  • Well-structured table with accurate cell values

Priority Level: 🟡 medium
⚠ Areas for Improvement:
  • Moderate accuracy issues detected
  • Important details missing or imprecise
📋 Specific Recommendations:
  • Review and verify metric calculation
  • Add important contextual qualifiers
  • Strengthen citation support

GENDER BIAS EVALUATION:

Bias Score: 0.00
Key Findings:
- The text does not make any references to gender, either in terms of individuals or professional roles.
Recommendations:
- Consider including examples or case studies that involve diverse individuals or teams in the field of quantum computing.
Detailed Statistics:
{
  "gender_distribution": {
    "male_references": 0,
    "female_references": 0
  },
  "professional_roles": {
    "male_associations": 0,
    "female_associations": 0
  }
}

<------------------------------------------------------------>

```

This script
