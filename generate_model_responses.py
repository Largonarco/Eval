import numpy as np
import json
import concurrent.futures
import copy

from api_utils import run_payload
from default_payload import DEFAULT_PAYLOAD


MAX_WORKERS = 8


def build_payload(user_query: str) -> dict:
    payload = copy.deepcopy(DEFAULT_PAYLOAD)
    payload["user_query"] = user_query
    return payload


def sample_queries(seed: int = 0, n_samples_per_subject: int = 1) -> list[str]:
    np.random.seed(seed)
    sampled_queries = []
    queries_by_subject = json.load(open("./example_queries.json"))
    for subject, subject_queries in queries_by_subject.items():
        sampled_query = np.random.choice(
            list(subject_queries), size=n_samples_per_subject, replace=False
        )[0]
        sampled_queries.append(sampled_query)
        print(f'Sampled "{sampled_query}" for subject "{subject}"')
    return sampled_queries


def generate_responses(queries: list[str]) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        worker_tasks = [
            executor.submit(run_payload, build_payload(query)) for query in queries
        ]
        concurrent.futures.wait(worker_tasks)
    models_responses = [worker_task.result() for worker_task in worker_tasks]
    with open("./model_responses.json", "w+") as f:
        print(json.dumps(models_responses, indent=4), file=f)


if __name__ == "__main__":
    queries = sample_queries()
    generate_responses(queries)
