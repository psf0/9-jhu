source ~/.bashrc
reference_dir=${1}
degraded_dir=${2}
output_path=${3}
matlab -nodisplay -nodesktop -r "try; addpath('src/external/'); addpath('src/evaluation/'); evaluate_folders('${reference_dir}', '${degraded_dir}', '${output_path}') ,catch e;fprintf(1,'\n%s',e.getReport); end; quit"
