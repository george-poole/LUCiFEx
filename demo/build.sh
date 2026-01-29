GLOB=$1
BUILD=$2
REMOTE=false

while [[ "$1" == --* ]]; do
    case "$1" in
        --remote)
        REMOTE=true
        shift
        ;;
    esac
done

IPYNB=($(find . -name "$GLOB.ipynb" -path "./notebooks/*"))
for i in "${IPYNB[@]}"
    do 
        echo Found notebook to execute $i
    done

for i in "${IPYNB[@]}"
    do 
        echo Executing $i 
        export IPYNB_FILE_NAME="${i}"
        jupyter nbconvert --execute --to notebook --inplace "${i}" --allow-errors  
    done

jupyter-book build . $BUILD

if $REMOTE; then
    ghp-import -n -p -f ./_build/html
fi