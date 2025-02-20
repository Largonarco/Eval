# Bias Evaluation (Gender)

I chose to tackle the problem of Gender Bias in LLM outputs.
I came up with a simple Gender Evaluator that assumes there are only 2 genders as I feel the outputs majority of times comprise of only the male and female gender mentions.

### Existing Setup

I went about creating an Bias Evaluator by doing some simple checks as follows

1. Only the pragraph blocks are analysed as they consist of the relevant content.
2. The blocks are then analysed for gendered terms/pronouns and also entity-pronoun relationships (where a pronoun is referring to a person) as these are referencing an entity they shouldn't count as bias. Spacy an NLP library is used to identify token dependencies.
   - Entities are extracted from the text first and dependency parsing is utilised in identifying if a known gendered pronoun refers to an entity.
3. After this the sentences are checked for pre-specified neutral professional terms and if those terms are surrounded by gendered pronouns implying an implicit bias
4. Based on the findings I am calculating bias metrics and providingg actionable feedback.

The dependency parsing logic works good but not in all cases. Simple structured structure are evaluated better compared to complex structured sentences. Although I feel with a bit fine tuning to the logic it could be made to perform better.

### Future Setup

I would go for an open-source LLM evaluation approach as embedding models would identify sentence structures and word co-relations much better, thus identifying type of bias appropriately.

1. Would setup a set of open-source LLMs, for example 3 models.
2. Would provide a structured prompt to all three models in parallel to identify gender bias and generate a score.
3. Average scores of all three models to mitigate evaluator LLM bias.

# Accuracy Evaluation (Metric and Tables)

### Existing Setup

Generated a systemmatic metric which helps better understand the accuracy and correctness of information given citation data.

1. In the first step basic checks are done to see if the metric and the table structure is proper, also if any citations are present they are extracted (not available to me).
2. In the next step both metric and table blocks are sent to OpenAI API along with a structured prompt to calculate a numeric score of accuracy (G-Eval)
3. If the accuracy score is below a certain threshold, intervention strategy is adopted by asking OpenAI API to suggest corrections given the citation data, alongside existing metric or table data.

### Future Setup

- Would make use of open-source LLMs to evaluate accuracy score based on citation data. Cost Effective and still efficient
- For intervention, would use LLMs which are connected to the web (Perplexity API) for better and correct corrections incase citation data is lacking.

# Relevance and Coherence Evaluation

### Existing Setup

Relevance and coherence are important aspects, as LLM's often have a tendency to hallucinate.

1. Took the entire response (set of blocks) along with the input query and specific evaluation steps to OpenAI API in-order to evaluate both relvance and coherence seperately.
2. Relied on inter block coherence and entire response's relevance to the user query to calculate both of the metrics seperately.
3. Made use of a detailed prompt so as to provide the LLM a methodlogicsl approach to taget the problem along with a strict scoring rubric.
4. Used asynchronous flow to make the pipeline non-blocking.

### Future Setup

- Use of open-source LLMs and parallel processing.
- Would incorporate human feedback and expand the golden dataset of preferred answers.
- Use this modified golden dataset to evaluate future answers and generate an evalutaion based on different sections of questions.
