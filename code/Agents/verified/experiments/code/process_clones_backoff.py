import sys
import os
import glob
import zipfile
import re
import json
import shutil
import time
import random
import string
import subprocess
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import tenacity
import psutil
import io
from typing import List, Tuple

class Shell:
    def __init__(
        self,
        shell_exec: bool = True,
        print_out: bool = True,
        print_cmd: bool = True,
        print_file: io.TextIOWrapper | None = None,
        return_list: bool = False,
    ) -> None:
        self.shell_exec = shell_exec
        self.print_out = print_out
        self.print_cmd = print_cmd
        self.print_file = print_file
        self.return_list = return_list

    def run(self, cmd: str | List[str], timeout: float | None = None) -> Tuple[str | List[str], str | List[str], int]:
        with subprocess.Popen(
            cmd, shell=self.shell_exec, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ) as p:
            if self.print_cmd:
                print(f'+ {cmd}', file=sys.stderr, flush=True)
                if self.print_file:
                    print(f'+ {cmd}', file=self.print_file, flush=True)
            out, err = [], []
            for line in iter(p.stdout.readline, ''):
                out.append(line)
                if self.print_out:
                    print(line, end='', flush=True)
            for line in iter(p.stderr.readline, ''):
                err.append(line)
                if self.print_out:
                    print(line, end='', flush=True, file=sys.stderr)
            p.wait()
            return ''.join(out), ''.join(err), p.returncode


@tenacity.retry(
    stop=tenacity.stop_after_attempt(8),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=64),
    retry=tenacity.retry_if_exception_type(RuntimeError),
)
def run_codeguru_inner(valid_scan_name, zip_file, region):
    path_json = f"{valid_scan_name}.json"
    if os.path.exists(path_json):
        os.remove(path_json)
    cmd = f'bash run_codeguru_security.sh {valid_scan_name} {zip_file} {region}'
    stdout, stderr, returncode = Shell().run(cmd)
    allout = stdout + stderr
    errmsg_throttling = 'An error occurred (ThrottlingException, TooManyRequestsException)'
    if errmsg_throttling in allout:
        raise RuntimeError(f'ThrottlingException: {allout}')


def run_codeguru(valid_scan_name, zip_file, region, error_log_folder):
    try:
        run_codeguru_inner(valid_scan_name, zip_file, region)
    except Exception as e:
        os.makedirs(error_log_folder, exist_ok=True)
        log_path = os.path.join(error_log_folder, f'err_{valid_scan_name}.log')
        with open(log_path, 'w') as f:
            f.write(str(e))
        print(f"Error logged in {log_path}")


def generate_random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def prepare_files(input_folder, output_folder, mode):
    os.makedirs(output_folder, exist_ok=True)
    metadata = {}

    for py_file in glob.glob(os.path.join(input_folder, "*.py")):
        if os.path.isfile(py_file):
            base_name = os.path.basename(py_file)
            new_name = base_name

            if mode == "codeguru":
                new_name = generate_random_string() + "_" + base_name

            new_file_path = os.path.join(output_folder, new_name)
            shutil.copy(py_file, new_file_path)

            metadata[py_file] = new_file_path

    meta_json_path = os.path.join(output_folder, "metadata.json")
    with open(meta_json_path, 'w') as meta_file:
        json.dump(metadata, meta_file, indent=4)

    return metadata


def split_files(metadata, num_splits=10):
    file_paths = list(metadata.values())
    random.shuffle(file_paths)  # Shuffle to distribute evenly

    chunks = [[] for _ in range(num_splits)]
    for i, file_path in enumerate(file_paths):
        chunks[i % num_splits].append(file_path)

    return chunks


def zip_files(file_list, output_folder):
    zip_files = []
    for file_path in file_list:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        zip_file = os.path.join(output_folder, f"{base_name}.zip")

        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(file_path, arcname=os.path.basename(file_path))  # Correct way to add files
        
        zip_files.append((base_name, zip_file))
    
    return zip_files


def main():
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <mode> <input_folder> <error_log_folder>")
        sys.exit(1)

    filtered_data = load_and_filter_results("evaluation_results.json")

    if not filtered_data:
        print("Error: No resolved instances found.")
    else:
        print(f"Processing {len(filtered_data)} resolved instances...")
        #clone_detection_results = process_clones(filtered_data)
        #save_responses(clone_detection_results)
        process_clones(filtered_data, output_dir="individual_results")

    mode = sys.argv[1]
    input_folder = sys.argv[2]
    error_log_folder = sys.argv[3]

    if mode not in ["vul", "codeguru", "fixes", "oss", "evol"]:
        print(f"Invalid mode: {mode}")
        sys.exit(1)

    output_folder = f"{input_folder}_{mode}_processed"
    metadata = prepare_files(input_folder, output_folder, mode)
    file_chunks = split_files(metadata)

    # Define 10 different AWS regions
    aws_regions = [
        "us-east-1", "us-east-2", "us-west-1", "us-west-2",
        "eu-west-1", "eu-west-2", "eu-central-1", "eu-north-1",
        "ap-southeast-1", "ap-southeast-2"
    ]

    tasks = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i, file_list in enumerate(file_chunks):
            region = aws_regions[i]  # Assign a different region to each chunk
            zip_file_list = zip_files(file_list, output_folder)  # Zip each file in the chunk

            for base_name, zip_file in zip_file_list:
                print(f"Running CodeGuru Security on {zip_file} with region {region}...")
                future = executor.submit(run_codeguru, base_name, zip_file, region, error_log_folder)
                tasks.append(future)
                time.sleep(0.55)

        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred while processing a scan: {e}")

    print("All scans are complete.")


if __name__ == '__main__':
    main()
