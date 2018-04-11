source ~/.bashrc
matlab -nodisplay -nosplash <<EOF
maxNumCompThreads(1);

base_path = '/export/b16/pfredericks/AAU/9-jhu/';
addpath([base_path 'src/external/']);

CHiME3_simulate_data_patched_parallel(true,20,'/export/corpora4/CHiME4/CHiME3','/export/corpora5/CHiME3')
EOF

