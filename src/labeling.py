# This code is based on the original repository by Junyi Ye (MIT License, 2024)
# Source: https://github.com/JunyiYe/CreativeMath

import argparse
import logging
import os
from datetime import datetime
from tqdm import tqdm
from logger import setup_logger
from models import ModelWrapper
from prompts import (load_coarse_grained_novelty_evaluation_prompt,
                     load_correctness_evaluation_prompt,)
from utils import extract_yes_no, load_json, save_json


evaluators = ["gemini-1.5-pro", "o4-mini"]

model_version = {
    "gemini-1.5-pro": "models/gemini-1.5-pro-002",
    "o4-mini": "o4-mini-2025-04-16"
}


def main():
    parser = argparse.ArgumentParser(description="Run the evaluation program.")
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--dataset_name", type=str, default="REF", help="REF, TEST, AIME, AHSME, AMC")
    parser.add_argument("--data_dir", type=str, default="data", help="dataset dir")
    parser.add_argument("--gen_model_name", type=str, default="Deepseek-math-7b-rl", help="The model was used in the experiment and will be evaluated.")
    parser.add_argument("--log_dir", type=str, default="logs", help="logging log dir")
    parser.add_argument("--log_level", type=str, default="INFO", help="logging log level")
    parser.add_argument("--data_dir", type=str, default="data", help="dataset dir")
    parser.add_argument("--output_dir", type=str, default="output", help="output dir")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature setting")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="maximum new generation tokens")
    
    
    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, args.dataset_name, 'labeling')
    os.makedirs(log_dir, exist_ok=True)

    output_dir = os.path.join(args.output_dir, args.dataset_name, 'labeling')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.gen_model_name}_labeled.json")


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{args.gen_model_name}_{timestamp}.log")
    logger = setup_logger(log_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Labeling the generated solutions on {args.dataset_name}")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")
    logger.warning("Ensure all transition sentences and justifications explaining the uniqueness of new solutions are manually removed to avoid influencing evaluator judgment. These sentences are usually at the beginning or ending of the response.")


    if args.dataset_name in ["REF", "TEST", "AMC, AIME", "AHSME"]:
        data_file = os.path.join(args.data_dir, f"{args.dataset_name}.json")
        dataset = load_json(data_file)
        
        pro_sol = {}
        for data in dataset:
            data_key = f'{data["competition_id"]}_{data["problem_id"]}'
            pro_sol[data_key] = {"problem": data["problem"],
                                 "solutions": data["solutions"]}

        gen_file = os.path.join(args.output_dir, args.dataset_name, "generation", f"{args.gen_model_name}.json")

    else:
        msg = f"Invalid dataset name '{args.dataset_name}'. Please choose from ['REF', 'TEST', 'AMC, AIME', 'AHSME']."
        logger.error(msg)
        raise ValueError(msg)


    # Evaluation file exists. Continuing unfinished evaluation.
    if os.path.exists(output_file):
        results = load_json(output_file)
    # Create the evaluation file and copy the experiment results.
    else:
        results = load_json(gen_file)
        for sample_id, sample in enumerate(results):
            results[sample_id]["labeling"]["correctness"] = {}
            results[sample_id]["labeling"]["novelty"] = {}


    for model_name in evaluators:
        args.model_name = model_name
        args.model_id = model_version[model_name]
        model = ModelWrapper(args)
    
        for sample_id, sample in tqdm(enumerate(results)):
            if model_name in sample["labeling"]["correctness"]:
                save_json(results, output_file)
                continue
            
            # Load problem and all solutions
            data_key = f'{sample["competition_id"]}_{sample["problem_id"]}'
            problem = pro_sol[data_key]["problem"]
            solutions = list(pro_sol[data_key]["solutions"].values())
            k = sample['k']

            # Load the generated new solution
            new_solution = sample["response"]

            prompt = load_correctness_evaluation_prompt(problem, solutions, new_solution)
            response = model.generate_response(prompt)
        
            decision = extract_yes_no(response)  # Return either "YES" or "NO"
            sample["labeling"]["correctness"][model_name] = decision

            if sample["labeling"]["correctness"][model_name] == "NO":
                sample["labeling"]["novelty"][model_name] = "NO"
                # If any of the LLM evaluators judges the answer as incorrect, label it as "Hallucinated_Solution"
                sample["label"] = "Hallucinated_Solution"
                results[sample_id] = sample
            else:
                prompt = load_coarse_grained_novelty_evaluation_prompt(problem, solutions, k, new_solution)
                response = model.generate_response(prompt)
                decision = extract_yes_no(response)  # Return either "YES" or "NO"
                sample["labeling"]["novelty"][model_name] = decision
                results[sample_id] = sample
            
            if sample_id % args.save_interval == 0:
                save_json(results, output_file)
        
        logger.info(f"Labeling results saved to {os.path.abspath(output_file)} using LLM-Evaluator {model_name}")
        save_json(results, output_file)
    
    for sample_id, sample in enumerate(results):
        if sample['label'] != "Hallucinated_Solution":
            novelty = sample["labeling"]["novelty"].values()
            yes_count = sum(1 for value in novelty if value == "YES")
            if yes_count == 0:
                sample["label"] = "Typical_Solution"
            else:
                sample["label"] = "Creative_Solution"

    logger.info(f"All results saved to {os.path.abspath(output_file)}")
    save_json(results, output_file)


    # Calculate accuarcy
    N = len(results)
    Hallucinated_count = 0
    Typical_count = 0
    Creative_count = 0

    for sample in results:
        final_res = sample["label"]
        if final_res == "Hallucinated_Solution":
            Hallucinated_count += 1
        elif final_res == "Typical_Solution":
            Typical_count += 1
        else:
            Creative_count += 1

    # 3-class
    Hallucinated_ratio = Hallucinated_count / N
    Typical_ratio = Typical_count / N
    Creative_ratio = Creative_count / N

    # 2-class
    correctness_count = Typical_count+Creative_count
    correctness_ratio = correctness_count / N

    if correctness_count != 0:
        Creative_to_correctness_ratio = Creative_count / correctness_count
    else:
        Creative_to_correctness_ratio = 0

    logger.info(f"The evaluation result for {args.dataset_name}_{args.gen_model_name} is as follows:")
    logger.info(f"Hallucinated Solution Ratio: {Hallucinated_ratio:.2%}")
    logger.info(f"Creative Solution Ratio: {Creative_ratio:.2%}")
    logger.info(f"Typical Solution Ratio: {Typical_ratio:.2%}")

    logger.info(f"Correctness Ratio: {correctness_ratio:.2%}")
    logger.info(f"Creative-to-Correctness Ratio: {Creative_to_correctness_ratio:.2%}")
    

if __name__ == "__main__":
    main()
