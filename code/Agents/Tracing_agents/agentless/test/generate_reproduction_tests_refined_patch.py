import argparse
import ast
import concurrent.futures
import json
import os
import re
from collections import Counter
from threading import Lock
import time #ira modified

from datasets import load_dataset
from tqdm import tqdm

from agentless.util.api_requests import num_tokens_from_messages
from agentless.util.model import make_model
from agentless.util.postprocess_data import remove_comments_and_docstrings
from agentless.util.utils import load_jsonl, setup_logger

generate_tests_prompt_template = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

The following patch has been proposed to solve the issue:
--- BEGIN PATCH ---
{proposed_patch}
--- END PATCH ---

Please generate a complete test that can be used to reproduce the issue.

The complete test should contain the following:
1. Necessary imports
2. Code to reproduce the issue described in the issue text
3. Print "Issue reproduced" if the outcome indicates that the issue is reproduced
4. Print "Issue resolved" if the outcome indicates that the issue has been successfully resolved
5. Print "Other issues" if the outcome indicates there are other issues with the source code

Here is an example:

```python
from sqlfluff import lint

def test__rules__std_L060_raised() -> None:
    try:
        sql = "SELECT   IFNULL(NULL, 100),
            NVL(NULL,100);"
        result = lint(sql, rules=["L060"])
        assert len(result) == 2
    except:
        print("Other issues")
        return

    try:
        assert result[0]["description"] == "Use 'COALESCE' instead of 'IFNULL'."
        assert result[1]["description"] == "Use 'COALESCE' instead of 'NVL'."
        print("Issue resolved")
    except AssertionError:
        print("Issue reproduced")
        return

    return

test__rules__std_L060_raised()
```

Please ensure the generated test reflects the issue described in the provided issue text.
The generated test should be able to be used to both reproduce the issue as well as to verify the issue has been fixed.
Wrap the complete test in ```python...```.
"""


def create_patch_from_code(python_code: str) -> str:
    patch_header = """diff --git a/reproduce_bug.py b/reproduce_bug.py
new file mode 100644
index 0000000..e69de29
"""
    patch_body = []
    patch_body.append("--- /dev/null")
    patch_body.append("+++ b/reproduce_bug.py")

    code_lines = python_code.split("\n")
    patch_body.append(f"@@ -0,0 +1,{len(code_lines)} @@")

    for line in code_lines:
        patch_body.append(f"+{line}")

    return patch_header + "\n".join(patch_body) + "\n"


def extract_first_code_block(text):
    pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    match = pattern.search(text)

    if match:
        return match.group(1).strip()

    return None


def gen_test(instance_id, args, swe_bench_data, prev_o, patch_lookup, write_lock=None):

    if args.target_id is not None:
        if args.target_id != instance_id:
            return

    log_file = os.path.join(
        args.output_folder, "generating_test_logs", f"{instance_id}.log"
    )
    logger = setup_logger(log_file)
    found = False
    for o in prev_o:
        if o["instance_id"] == instance_id:
            found = True
            break

    if found:
        logger.info(f"skipping {instance_id} since patch already generated")
        return None

    logger.info(f"================ generating test for {instance_id} ================")

    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]

    # extra_info = extra_info_lookup.get(instance_id, {})

    # # Extract found_files
    # found_files = extra_info.get("found_files", [])

    # # Extract found_related_locs
    # found_related_locs = []
    # for file, locs in extra_info.get("found_related_locs", {}).items():
    #     for loc in locs:
    #         # split locs by newline if needed
    #         found_related_locs.extend(loc.strip() for loc in loc.split("\n") if loc.strip())

    # # Extract found_edit_locs
    # found_edit_locs = []
    # for edit_entry in extra_info.get("found_edit_locs", []):
    #     for file, locations in edit_entry.items():
    #         for loc in locations:
    #             if loc.strip():
    #                 found_edit_locs.append(f"{file}: {loc}")
    # found_edit_locs = list(set(found_edit_locs))  # deduplicate

    proposed_patch_entry = patch_lookup.get(instance_id, {})

    if proposed_patch_entry:
        proposed_patch = proposed_patch_entry.get("model_patch", "").strip()
    else:
        proposed_patch = ""
    

    raw_outputs, counts, all_generations, traj = (
        [],
        [],
        [],
        [],
    )

    raw_output = ""

    prompt_template = generate_tests_prompt_template

    # message = prompt_template.format(
    #     problem_statement=problem_statement,
    # ).strip()


    message = prompt_template.format(
                problem_statement=problem_statement,
                proposed_patch=proposed_patch,
    ).strip()


    logger.info(f"prompting with message:\n{message}")

    all_generations, counts, traj = [], [], []
    sample_responses = []

    # get greedy sample
    model = make_model(
        model=args.model,
        logger=logger,
        backend=args.backend,
        max_tokens=1024,
        temperature=0,
        batch_size=1,
    )
    if args.skip_greedy:
        greedy_traj = {
            "response": "",
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": 0,
            },
        }
    else:
        if args.mock:
            greedy_traj = {
                "response": "",
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, args.model),
                },
            }
        else:
            greedy_traj = model.codegen(
                message, num_samples=1, prompt_cache=args.max_samples > 1
            )[0]

    sample_responses.append(greedy_traj)
    # get temperature samples

    sample_trajs = []           # Ira modified

    model = make_model(
        model=args.model,
        logger=logger,
        backend=args.backend,
        max_tokens=1024,
        temperature=0.8, #0.8 default #1.0 first exp #0.2 second exp #0.6 thrd exp
        batch_size=1 if args.backend == "anthropic"           # <<< change
               else args.max_samples - 1, #Ira modified # batch_size=args.max_samples - 1,  # minus the 1 greedy sample
    )

    if args.mock:
        first_traj = {
            "response": "",
            "usage": {
                "prompt_tokens": num_tokens_from_messages(message, args.model),
            },
        }
        later_traj = {
            "response": "",
            "usage": {"prompt_tokens": 0},
        }
        if args.max_samples - 1:
            sample_trajs = [first_traj] + [later_traj] * (args.max_samples - 2)
        else:
            sample_trajs = []
    elif args.backend == "anthropic" and args.max_samples - 1: #Ira modified
        #######################################################################
        # Anthropic: parallel-but-throttled loop to stay under 8 000 otp/min   #
        #######################################################################
        MAX_TOKENS_PER_REQ = 1024               # safe upper-bound for one reply
        TOKENS_PER_MIN     = 8000              # Build-tier default
        CHUNK = min(args.num_threads,          # parallel calls you asked for
                    TOKENS_PER_MIN // MAX_TOKENS_PER_REQ or 1)

        # Re-build the model so max_tokens == our cap
        model = make_model(
            model=args.model,
            logger=logger,
            backend=args.backend,
            max_tokens=MAX_TOKENS_PER_REQ,
            temperature=0.8,
            batch_size=1,                      # Anthropic requires 1
        )

        need = args.max_samples - 1            # already have the greedy sample
        sample_trajs = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=CHUNK) as pool:
            while need:
                now  = min(CHUNK, need)        # how many to launch this round
                futs = [
                    pool.submit(model.codegen,
                                message,
                                num_samples=1, # MUST be 1 with Anthropic
                                prompt_cache=True)
                    for _ in range(now)
                ]
                for f in concurrent.futures.as_completed(futs):
                    print("Extending trajs...")
                    sample_trajs.extend(f.result())

                need -= now

                # If we still have more to fetch, wait a full minute so
                #   CHUNK × MAX_TOKENS_PER_REQ ≤ 8 000 tokens per minute.
                if need:
                    print("Time...")
                    time.sleep(60)
    else:
        if args.max_samples - 1:
            # always use cached prompt if possible for later samples
            sample_trajs = model.codegen(
                message, num_samples=args.max_samples - 1, prompt_cache=True
            )
        else:
            sample_trajs = []

    sample_responses.extend(sample_trajs)

    count = 0
    while count < args.max_samples:
        print(f"trying the {count + 1}-th sample ...")
        ret = sample_responses[count]
        count += 1
        traj.append({**ret, "prompt": message})

        if args.mock:
            continue

        raw_output = ret["response"]
        logger.info(f"raw output:\n{raw_output}")
        print((f"raw output:\n{raw_output}"))
        all_generations.append(raw_output)

        counts.append(count)
        raw_outputs.append(raw_output)

    if write_lock is not None:
        write_lock.acquire()
    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "raw_output": raw_outputs,
                    "all_generations": [all_generations],
                    "try_count": counts,
                    "traj": traj,
                    "prev_content": [
                        [""]
                    ],  # To make the tests compatible with the repair setup
                    "file_names": [["reproduce_bug.py"]],
                }
            )
            + "\n"
        )
    if write_lock is not None:
        write_lock.release()


def generate_tests(args):
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    
    patch_data = load_jsonl(args.proposed_patch_file)
    patch_lookup = {entry["instance_id"]: entry for entry in patch_data}

    swe_bench_data = load_dataset(args.dataset, split="test")
    instances = swe_bench_data["instance_id"]
    prev_o = load_jsonl(args.output_file) if os.path.exists(args.output_file) else []

    if args.num_threads == 1:
        for instance_id in tqdm(instances, total=len(instances), colour="MAGENTA"):
            gen_test(instance_id, args, swe_bench_data, prev_o, patch_lookup)
    else:
        write_lock = Lock()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads
        ) as executor:
            futures = {
                executor.submit(
                    gen_test, instance_id, args, swe_bench_data, prev_o, patch_lookup, write_lock,
                ): instance_id
                for instance_id in instances
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(instances),
                colour="MAGENTA",
            ):
                future.result()


def post_process_tests(args):
    """
    apply some diff formatting.
    """
    raw_outputs = load_jsonl(args.raw_output_file)
    generation_idx = args.select_id

    for raw_output in raw_outputs:
        print("Raw output =...")
        instance_id = raw_output["instance_id"]

        if (
            raw_output["raw_output"] == ""
            or not raw_output["all_generations"][0][generation_idx]
        ):
            with open(args.output_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "model_name_or_path": "agentless",
                            "instance_id": instance_id,
                            "test_patch": "",
                        }
                    )
                    + "\n"
                )
            continue

        if args.select_id == -1:
            # Use the last generation
            assert False, "not implemented for now"
        else:
            print("Got to extacting code block....")
            raw_git_diffs = raw_output["all_generations"][0][generation_idx]
            #print(raw_git_diffs)
            extracted_code = extract_first_code_block(
                raw_output["all_generations"][0][generation_idx]
            )
            print(extracted_code)
            if extracted_code:
                print("Extracted code!!!")
                git_diffs = create_patch_from_code(extracted_code)
            else:
                git_diffs = ""


        with open(args.output_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "model_name_or_path": "agentless",
                        "instance_id": instance_id,
                        "test_patch": git_diffs.lstrip(),
                        "raw_test_patch": raw_git_diffs,
                        "original_file_content": "",
                    }
                )
                + "\n"
            )


def normalize_test(test: str):
    def normalize_code(code):
        try:
            node = ast.parse(code)
            return ast.unparse(node)
        except:
            return code

    test = normalize_code(test)

    try:
        remove_docstring_test = remove_comments_and_docstrings(test)
        ast.parse(remove_docstring_test)  # check
    except:
        remove_docstring_test = test

    try:
        test_name = remove_docstring_test.splitlines()[-1].split("(")[0]
        remove_docstring_test = remove_docstring_test.replace(
            test_name, "test_func"
        )  # use generic name
    except:
        pass

    return remove_docstring_test


def normalize_tests(args):

    selected_ids = list(range(int(args.max_samples)))

    for i in selected_ids:
        if os.path.exists(
            f"{args.output_folder}/output_{i}_normalized_reproduction_test.jsonl"
        ):
            continue

        tests = load_jsonl(
            f"{args.output_folder}/output_{i}_processed_reproduction_test.jsonl"
        )
        for d in tests:
            test = extract_first_code_block(d["raw_test_patch"])
            normalized_test = normalize_test(test)
            d["normalized_test"] = normalized_test

        with open(
            f"{args.output_folder}/output_{i}_normalized_reproduction_test.jsonl", "w"
        ) as f:
            for d in tests:
                f.write(json.dumps(d) + "\n")


def get_sample(execution_results, instance_id, sample_id) -> tuple[str, bool]:
    """Returns the diff and pass status."""
    return execution_results[instance_id][sample_id]


def test_selection(args):
    from pathlib import Path

    test_exec_results = {}

    roots = [Path(folder) for folder in args.output_folder.split(",")]

    intervals = [(0, int(args.max_samples / len(roots)) - 1) for _ in range(len(roots))]

    interval = intervals[0]
    for i in range(interval[0], interval[1] + 1):
        for index, root in enumerate(roots):
            test_patches = load_jsonl(
                f"{root}/output_{i}_normalized_reproduction_test.jsonl"
            )
            filtered_results = load_jsonl(
                f"{root}/output_{i}_processed_reproduction_test_verified.jsonl"
            )
            print(
                f"Loaded {len(test_patches)} patches from {root}/output_{i}_normalized.jsonl"
            )
            for test_patch in test_patches[:]:
                filtered = [
                    x
                    for x in filtered_results
                    if x["instance_id"] == test_patch["instance_id"]
                ]
                if len(filtered) != 0:
                    filtered_status = filtered[0]["verified"]
                else:
                    filtered_status = False
                test_exec_results.setdefault(test_patch["instance_id"], []).append(
                    {
                        "normalized_test": test_patch["normalized_test"].strip(),
                        "test_patch": test_patch["test_patch"],
                        "filtered_status": filtered_status,
                    }
                )

    total_count = 0

    for instance_id in test_exec_results:
        patch_keys = [
            test_exec_results[instance_id][i]["normalized_test"]
            for i in range(len(test_exec_results[instance_id]))
        ]
        verified = [
            test_exec_results[instance_id][i]["filtered_status"]
            for i in range(len(test_exec_results[instance_id]))
        ]

        patch_ids = [
            i
            for i in range(len(test_exec_results[instance_id]))
            if patch_keys[i].strip() and verified[i]
        ]

        if not patch_ids:
            continue

        vote = Counter()
        first_appear_idx = dict()
        unique_patches = set()
        for i in patch_ids:
            sample = get_sample(test_exec_results, instance_id, i)
            patch_key, patch = sample["normalized_test"], sample["test_patch"]
            vote[patch_key] += 1
            if patch_key not in first_appear_idx:
                first_appear_idx[patch_key] = i

            unique_patches.add(patch_key)

        maj_selected_id = max(
            patch_ids,
            key=lambda i: (vote[patch_keys[i]], -first_appear_idx[patch_keys[i]]),
        )

        print([vote[patch_keys[i]] for i in patch_ids])

        sample = get_sample(test_exec_results, instance_id, maj_selected_id)

        result = {
            "model_name_or_path": "agentless",
            "instance_id": instance_id,
            "test_patch": sample["test_patch"],
        }

        total_count += 1

        with open(f"{args.output_folder}/{args.output_file}", "a") as f:
            f.write(json.dumps(result) + "\n")

    print(total_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=20, help="Sampling budget.")
    parser.add_argument(
                        "--proposed_patch_file",
                        type=str,
                        required=True,
                        help="Path to JSONL file containing the model_patchs for each instance",
    )
    parser.add_argument(
        "--select_id",
        type=int,
        default=-1,
        help="Index the selected samples during post-processing.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=[
            "gpt-4o-2024-05-13",
            "deepseek-coder",
            "gpt-4o-mini-2024-07-18",
            "claude-3-5-sonnet-20241022",
        ],
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "deepseek", "anthropic"],
    )
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--skip_greedy", action="store_true")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )
    parser.add_argument("--target_id", type=str)
    parser.add_argument(
        "--mock", action="store_true", help="Mock run to compute prompt tokens."
    )
    parser.add_argument("--select", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Verified",
        choices=["princeton-nlp/SWE-bench_Lite", "princeton-nlp/SWE-bench_Verified"],
    )

    args = parser.parse_args()

    assert (not "deepseek" in args.model) or (
        args.backend == "deepseek"
    ), "Must specify `--backend deepseek` if using a DeepSeek model"

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if not os.path.exists(os.path.join(args.output_folder, "generating_test_logs")):
        os.makedirs(os.path.join(args.output_folder, "generating_test_logs"))

    if not args.select:
        args.output_file = os.path.join(args.output_folder, "output.jsonl")
        generate_tests(args)
        args.raw_output_file = args.output_file
        for i in range(args.max_samples):
            args.output_file = args.raw_output_file.replace(
                ".jsonl", f"_{i}_processed_reproduction_test.jsonl"
            )
            args.select_id = i
            post_process_tests(args)
    else:
        normalize_tests(args)
        test_selection(args)


if __name__ == "__main__":
    main()
