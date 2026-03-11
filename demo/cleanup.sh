REMOTE=false
DRY=false

while [[ "$1" == --* ]]; do
    case "$1" in
        --dry)
        DRY=true
        shift
        ;;
    esac
done

GLOB_FILE=$1
GLOB_DIR=${2:-'./**'}

EXTS=("pdf" "png" "mp4" "h5" "xdmf" "npy" "npz" "txt")
FILES=()

for ext in "${EXTS[@]}"
    do 
        FOUND=($(find . -name "$GLOB_FILE.$ext" -path "./$GLOB_DIR"))
        FILES=("${FILES[@]}" "${FOUND[@]}")
    done

for file in "${FILES[@]}"
    do 
        echo Found file to cleanup $file 
    done

if $DRY; then
    echo "Exiting dry run"
    exit
fi

for i in "${FILES[@]}"
    do 
        echo Cleaning up $i 
        rm $i
    done