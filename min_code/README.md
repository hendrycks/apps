# Evaluation

### First, generate the gpt code

    python3 generate_gpt_codes.py -t /path/to/apps-beta/test  --save /path/to/save_dir

### Then evaluate the accuracy of the outputted code

    python3 test_one_solution.py -t /path/to/apps-beta/test --save /path/to/save_dir
    # because the above may fail on account of poorly generated python programs 
    # we suggest to run a for loop for each problem index against the "all_codes.json"
    for i in {0..#Num_Problems#} ; do 
      python3 test_one_solution.py -t /path/to/apps-beta/test --save /path/to/save_dir -i $i ;
    done
