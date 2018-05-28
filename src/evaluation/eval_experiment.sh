
#clean epochs gpu

#naming experiment_name _ i
experiment_name=$1

for i in 7 9 10 11 21 22 #epochs
do
experiment_name_i=${experiment_name}_${i}

#spawn Pesq and matlab
fs=8000
reference_dir=data/processed/dataset_4/val/x
degraded_dir=data/interim/dataset_4_val/${experiment_name_i}
output_path=models/${experiment_name_i}/PESQ_results.mat
output_path_matlab=models/${experiment_name_i}/eval_results.mat

echo qsub -e ERRPESQ -o LOGPESQ -cwd -l mem_free=5G,ram_free=5G -pe smp 4 ./sub_cpu.sh src/evaluation/PESQ_evalfolder.py $fs $reference_dir $degraded_dir $output_path
qsub -e ERRPESQ -o LOGPESQ -cwd -l mem_free=5G,ram_free=5G -pe smp 4 ./sub_cpu.sh src/evaluation/PESQ_evalfolder.py $fs $reference_dir $degraded_dir $output_path

echo qsub -e ERRMATLAB -o LOGMATLAB -cwd -l mem_free=5G,ram_free=5G -pe smp 4 ./src/evaluation/matlab_eval.sh $reference_dir $degraded_dir $output_path_matlab
qsub -e ERRMATLAB -o LOGMATLAB -cwd -l mem_free=5G,ram_free=5G -pe smp 4 ./src/evaluation/matlab_eval.sh $reference_dir $degraded_dir $output_path_matlab

#Run LRE17 evaluation
lre17_dev_mls14_clean=../../data/interim/lre17tel_dev/${experiment_name_i}
lre17_dev_vast_clean=../../data/interim/lre17_dev/${experiment_name_i}
lre17_eval_mls14_clean=../../data/interim/lre17tel_eval/${experiment_name_i}
lre17_eval_vast_clean=../../data/interim/lre17_eval/${experiment_name_i}
cd lre17/v1.d/
echo ./run_cleaned_tool.sh ${experiment_name_i} $lre17_dev_mls14_clean $lre17_dev_vast_clean
./run_cleaned_tool.sh ${experiment_name_i} $lre17_dev_mls14_clean $lre17_dev_vast_clean $lre17_eval_mls14_clean $lre17_eval_vast_clean
cd ../../
done
