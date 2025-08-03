#YOUR ENVIRONMENT

#datasets: [gsm8k, math, triviaqa, mmlu, mmlu_cot]
srun python dataset_creation2.py --model meta-llama/Meta-Llama-3-8B-Instruct --num_samples 7000 --seed 42 --dataset mmlu_cot