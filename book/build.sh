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

IPYNB_EXT="ipynb"
THUMBNAILS_DIR="thumbnails"
THUMBNAILS_EXT="png"
ERROR_KEYWORD="Traceback"

if [ -z "$TARGET_DIR" ]; then
    DIRS=("demo" "benchmark")
else
    DIRS=($TARGET_DIR)
fi

for dir in "${DIRS[@]}"; do
    unlink $dir
    ln -s "../$dir" $dir 
    ipynb_paths=($(find -L . -name "$GLOB.$IPYNB_EXT" -path "./$dir/*"))
    for ipynb in "${ipynb_paths[@]}"; do
        echo Found notebook to execute $ipynb
        if ! $DRY; then
            echo Executing notebook $ipynb
            ipynb_name=$(basename $ipynb ".$IPYNB_EXT")
            ipynb_dir=$(dirname $ipynb)
            rm "$ipynb_dir/$THUMBNAILS_DIR/$ipynb_name.$THUMBNAILS_EXT"
            echo Beginning execution "$(date)"
            export IPYNB_FILE_PATH="$ipynb"
            jupyter nbconvert --execute --to notebook --inplace $ipynb $NBCONVERT_ARGS  
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

for dir in "${DIRS[@]}"; do
    ipynb_paths=($(grep -rl $ERROR_KEYWORD ./$dir/* --include="*.$IPYNB_EXT"))
    for ipynb in "${ipynb_paths[@]}"; do
        echo ""
        echo "WARNING! Error found in $ipynb"
    echo ""
    done
done