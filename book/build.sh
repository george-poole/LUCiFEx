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
TARGET_DIR=${2:-""}
NBCONVERT_ARGS=${3:-"--allow-errors"}
BUILD_ARGS=${4:-""}

if [ -z "$TARGET_DIR" ]; then
    DIRS=("demo" "benchmark")
else
    DIRS=($TARGET_DIR)
fi

for dir in "${DIRS[@]}"; do
    unlink $dir
    ln -s "../$dir" $dir 
    IPYNB=($(find . -name "$GLOB.ipynb" -path "../$dir/*"))
    for i in "${IPYNB[@]}"; do
        echo Found notebook to execute $i
        if ! $DRY; then
            echo Executing notebook $i 
            export IPYNB_FILE_NAME="${i}"
            echo Beginning execution "$(date)"
            jupyter nbconvert --execute --to notebook --inplace "${i}" $NBCONVERT_ARGS  
            echo Finished execution "$(date)"
        fi
    done    
done

if $DRY; then
    echo "Exiting dry run"
    exit
fi

echo Making gallery ...
python make_gallery.py
echo Gallery made

jupyter-book build . $BUILD_ARGS
ln -sf "./_build/html/index.html" alias.html

if $REMOTE; then
    ghp-import -n -p -f ./_build/html
fi