
#clean epochs gpu

#naming experiment_name _ i
experiment_name=$1

for i in $(seq $2 $3)
do
experiment_name_i=${experiment_name}_${i}

#Run LRE17 evaluation
lre17_dev_mls14_clean=../../data/interim/lre17tel_dev/${experiment_name_i}
lre17_dev_vast_clean=../../data/interim/lre17_dev/${experiment_name_i}
cd lre17/v1.d2/
echo ./run_cleaned_dev_tool.sh ${experiment_name_i} $lre17_dev_mls14_clean $lre17_dev_vast_clean
./run_cleaned_dev_tool.sh ${experiment_name_i} $lre17_dev_mls14_clean $lre17_dev_vast_clean
cd ../../
done
