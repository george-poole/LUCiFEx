NOTEBOOK=${1:-0}

if [ $NOTEBOOK -eq 0 ]
then
    IPYNB=($(find . -name "*.ipynb"))
    echo "${IPYNB[@]}"
    for i in "${IPYNB[@]}"
    do 
        jupyter nbconvert --execute --to notebook --inplace "${i}" --allow-errors  
    done
else
   jupyter nbconvert --execute --to notebook --inplace $NOTEBOOK --allow-errors  
fi