GLOB=$1
IPYNB=($(find . -name "${GLOB}*.ipynb"))

for i in "${IPYNB[@]}"
    do 
        echo $i found
    done

echo 

for i in "${IPYNB[@]}"
    do 
        echo $i executing ...
        jupyter nbconvert --execute --to notebook --inplace "${i}" --allow-errors  
    done