#!/bin/bash
name=$1
week="week4"
if [ ! -z ${name} ]; then
    week="week4_${name}"
fi

main_directory=/home/grupo01/$week

results_directory=${main_directory}/results
for test_type in $(ls -l ${results_directory} | cut -d' ' -f9 | cut -d'_' -f1 | sort | uniq)
do
    headers=$(cat ${path}/config.ini | grep -v "DEFAULT" | cut -d '=' -f1 | paste -sd'|' -)
    echo "Test|ID|${headers}|ValAcc|ValLoss|TestAcc|OutputFile" | sed 's/\ //g'
    for directory in `ls  ${results_directory} | grep ${test_type} |cut -f9`
    do
        path=${results_directory}/${directory}
        output_file=$(find ${path} -type f -iname "*.out" | sort | tail -1)
        config_file=${path}/config.ini
        if [ ! -z ${output_file} ] && [ -f ${config_file} ]
        then
            config_values=$(cat ${path}/config.ini | grep -v "DEFAULT" | cut -d '=' -f2 | paste -sd'|' -)
            exec_id=$(basename ${output_file} | cut -d'_' -f1)        
            acc_loss=$(cat ${output_file} | grep val_loss | awk -F "[-:]" 'BEGIN { max_acc=0;loss=0; } {if($10>max_acc){max_acc=$10;loss=$8;}} END {print max_acc"|"loss}')
            test_val=$(cat ${output_file} | grep "Test Acc" | awk -F"=" '{print $2}')
            echo "${directory}|${exec_id}|${config_values}|${acc_loss}|${test_val}|${output_file}" | sed 's/\ //g'
        fi
    done
done