source ~/.bashrc
matlab -nodisplay -nodesktop -r "try; run('$@') ,catch e;fprintf(1,'\n%s',e.getReport); end; quit"

