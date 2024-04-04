num_workers=0

for seed in 1 #2 3 4 5 
do
  for model_type in coteaching_plus
  do
    CUDA_LAUNCH_BLOCKING=1 python main.py --dataset cicids --model_type ${model_type} --noise_type symmetric --noise_rate 0.1 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
    # CUDA_LAUNCH_BLOCKING=1 python main.py --dataset cicids --model_type ${model_type} --noise_type symmetric --noise_rate 0.2 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
    # CUDA_LAUNCH_BLOCKING=1 python main.py --dataset cicids --model_type ${model_type} --noise_type pairflip --noise_rate 0.3 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
  done
done
