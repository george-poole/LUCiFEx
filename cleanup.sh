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

GLOB=$1
NOTEBOOKS='notebooks'
FIGURES='figures'

PDFS=($(find . -name "$GLOB.pdf" -path "./${NOTEBOOKS}/**/${FIGURES}/*"))
PNGS=($(find . -name "$GLOB.png" -path "./${NOTEBOOKS}/**/${FIGURES}/*"))
MP4S=($(find . -name "$GLOB.mp4" -path "./${NOTEBOOKS}/**/${FIGURES}/*"))

FILES=("${PDFS[@]}" "${PNGS[@]}" "${MP4S[@]}")

for i in "${FILES[@]}"
    do 
        echo Found file to cleanup $i 
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