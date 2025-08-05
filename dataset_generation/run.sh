#YOUR ENVIRONMENT

#datasets: [gsm8k, math, triviaqa, mmlu, mmlu_cot]
python dataset_creation.py --model meta-llama/Meta-Llama-3-8B-Instruct --num_samples 10000 --seed 42 --dataset mmlu_cot

python process_dataset.py --model llama --dataset mmlu_cot --seed 42