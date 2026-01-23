GLOB=$1
IPYNB=($(find . -name "$GLOB.ipynb" -path "./notebooks/*"))

for i in "${IPYNB[@]}"
    do 
        echo Found notebook $i
    done

for i in "${IPYNB[@]}"
    do 
        echo Executing $i 
        export IPYNB_FILE_NAME="${i}"
        jupyter nbconvert --execute --to notebook --inplace "${i}" --allow-errors  
    done