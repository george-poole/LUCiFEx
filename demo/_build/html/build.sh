REMOTE=false
DRY=false

while [[ "$1" == --* ]]; do
    case "$1" in
        --remote)
        REMOTE=true
        shift
        ;;
    esac
    case "$1" in
        --dry)
        DRY=true
        shift
        ;;
    esac
done

GLOB=$1
BUILD=$2

IPYNB=($(find . -name "$GLOB.ipynb" -path "./notebooks/*"))
for i in "${IPYNB[@]}"
    do 
        echo Found notebook to execute $i
    done

if $DRY; then
    echo "Exiting dry run"
    exit
fi

for i in "${IPYNB[@]}"
    do 
        echo Executing notebook $i 
        export IPYNB_FILE_NAME="${i}"
        echo Beginning excution "$(date)"
        jupyter nbconvert --execute --to notebook --inplace "${i}" --allow-errors  
        echo Finished excution "$(date)"
    done

echo Building gallery ...
python build_gallery.py
echo Gallery built

jupyter-book build . $BUILD

if $REMOTE; then
    ghp-import -n -p -f ./_build/html
fi