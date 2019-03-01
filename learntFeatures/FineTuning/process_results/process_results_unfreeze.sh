#! /bin/bash
name=$1
week="week4"
if [ ! -z ${name} ]; then
    week="week4_${name}"
fi

main_directory=/home/grupo01/$week
results_directory=${main_directory}/results
for test_type in $(ls -l ${results_directory} | cut -d' ' -f9 | cut -d'_' -f1 | sort | uniq)
do

    headers_config_file=$(find ${results_directory} -type f -wholename "*${test_type}_*/config.ini" | head -1)
    if [ ! -z ${headers_config_file} ]
    then
        headers=$(cat ${headers_config_file} | grep -v "DEFAULT" | cut -d '=' -f1 | paste -sd'|' -)
        echo "${test_type}_Test|ID|${headers}|ValAcc_Before|ValLoss_Before|ValAcc_After|ValLoss_After|TestAcc|OutputFile" | sed 's/\ //g'
        for directory in `ls  ${results_directory} | grep ${test_type} |cut -f9`
        do
            path=${results_directory}/${directory}
            output_file=$(find ${path} -type f -iname "*.out" | sort | tail -1)
            # count lines file
            count=$(wc -l $output_file | cut -d " " -f 1)

            config_file=${path}/config.ini
            if [ ! -z ${output_file} ] && [ -f ${config_file} ]
            then
                config_values=$(cat ${path}/config.ini | grep -v "DEFAULT" | cut -d '=' -f2 | paste -sd'|' -)
                exec_id=$(basename ${output_file} | cut -d'_' -f1)

                # Before Defrost
                acc_loss_before=$(cat ${output_file} | grep -B $count 'Defrosted' | grep val_loss | awk -F "[-:]" 'BEGIN { max_acc=0;loss=0; } {if($10>max_acc){max_acc=$10;loss=$8;}} END {print max_acc"|"loss}')
                # After Defrost
                acc_loss_after=$(cat ${output_file} | grep -A $count 'Defrosted' | grep val_loss | awk -F "[-:]" 'BEGIN { max_acc=0;loss=0; } {if($10>max_acc){max_acc=$10;loss=$8;}} END {print max_acc"|"loss}')
                
                test_val=$(cat ${output_file} | grep "Test Acc" | awk -F"=" '{print $2}')
                echo "${directory}|${exec_id}|${config_values}|${acc_loss_before}|${acc_loss_after}|${test_val}|${output_file}" | sed 's/\ //g'
            fi
        done
    fi
done
