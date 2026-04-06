CONFIRM=false

while [[ "$1" == --* ]]; do
    case "$1" in
        --confirm)
        CONFIRM=true
        shift
        ;;
    esac
done

GLOB_FILE=$1
GLOB_DIR=${2:-'./**'}
TARGET_EXT=${3:-""}

if [ -z "$TARGET_EXT" ]; then
    EXTS=("pdf" "png" "mp4" "h5" "xdmf" "npy" "npz" "txt")
else
    EXTS=($TARGET_EXT)
fi

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

if ! $CONFIRM; then
    echo "Exiting unconfirmed run"
    exit
fi

for i in "${FILES[@]}"
    do 
        echo Purging $i 
        rm $i
    done