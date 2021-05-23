#!/bin/bash
#SBATCH -p high_pre 
#SBATCH -w balrog-cpu
#SBATCH -c 2 
#SBATCH --output=R-%j.out

echo $1 $2 $3 $4 $5
pushd .

cd $1
echo "python eval_bleu.py -s $2 -e $3 --save $4 --test_loc $5"
python eval_bleu.py -s $2 -e $3 --save $4 --test_loc $5 #--debug
echo "python test_one_solution.py -s $2 -e $3 --save $4 --test_loc $5"
python test_one_solution.py -s $2 -e $3 --save $4 --test_loc $5 #--debug

popd
mkdir slurm-output
mv R-${SLURM_JOB_ID}.out slurm-output

