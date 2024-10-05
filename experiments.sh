#!/bin/bash

# Last experiment = multiple runs
# first run with temp and top_p = 0
# second run with temp and top_p = 1
# The results here reflects table 2 in the paper
# and Figure 2.
temp=0.0
top_p=0.0
data=data/advbench/advbench_arabic_no_num_no_ar_letters.csv
for model in claude-3-sonnet-20240229 gpt-4 llama2
do
    for dialog in chatspeak transliteration pure_arabic english leetspeak
    do
        python llm-test-ar.py --dialog_type $dialog \
        --model $model --data $data \
        --temperature $temp --top_p $top_p \
        &> outputs/$model/advbench/$(dialog)_no_numbers_top-p_temp=$(temp).log

    temp=$((temp+1))
    top_p=$((top_p+0.9))
    done
done

# For Arabic with prefix "sure, here is "
for model in claude-3-sonnet-20240229 gpt-4
do
    python llm-test-ar.py --dialog_type pure_arabic \
    --model $model \
    --data data/advbench_arabic_no_numbers.csv \
    --prompt_injection_type suffix --prompt_injection_value "بكل تأكيد " \
    &> outputs/$model/advbench/pure_arabic_with_prefix_injection.log
done

# For mitigation experiments
# just pass --mitigate True
dialog=Transliteration
model=gpt-4
python llm-test-ar.py --dialog_type $dialog \
--model $model --data $data \
&> outputs/$model/advbench/$(dialog)_no_numbers_mitigation.log

# For the last experiment in table 6 appendix B
# first collect the copyright-answered prompt. and
# google related prompts. You find them in the data.
# check helper codes in translate_to_arabic to create
# advbench_En_Transliteration_copyright.csv if needed.
# Then run the following command: change temperature and top_p as needed.


# prompts noncopyright_refusal could be found in file:
# data/advbench/advbench_En_Transliteration_noncopyright_refusal.csv
# prompts with copyright answers could be found in file:
# advbench_En_Transliteration_copyright.csv
for data in data/advbench/advbench_En_Transliteration_noncopyright_refusal.csv advbench_En_Transliteration_copyright.csv
runs=1
do
    python llm-test-ar.py \
    --dialog_type transliteration \
    --model claude-3-sonnet-20240229 \
    --data $data \
    --temperature 0 --top_p 0 \
    --prompt_injection_type both \
    --prompt_injection_value "āktb skrybt_KAO" \
    &> outputs/claude-3-sonnet-20240229/advbench/transliteration_temp_top_p\=0_runs_$(runs).log
    runs=$((runs+1))
done

# For Google-answered prompts check files:
# nonperturbed_En_Ar_nongoogle_refusal.csv.
# arabic_only_google_perturbed_and_nonperturbed.csv. However,
# we perturbed them with the following command and the
# results are in files:
# perturbed_En_Ar_nongoogle_refusal.csv (for nongoogle), and
# arabic_only_google_perturbed_and_nonperturbed.csv (for google)


# perturbation command for google non-google prompts
python llm-test-ar.py \
--dialog_type pure_arabic \
--model gpt-4o \
--data data/advbench/nonperturbed_En_Ar_nongoogle_refusal.csv \
--temperature 0 --top_p 0 --perturb_sentence yes --perturb_output_path data/advbench/perturbed_En_Ar_nongoogle_refusal.csv

# After perturbation, change the dialog_type to pure_arabic_perturbed

