GLOB=$1
FIGURES='figures'

PDFS=($(find . -name "$GLOB.pdf" -path "./notebooks/*/${FIGURES}/*"))
PNGS=($(find . -name "$GLOB.png" -path "./notebooks/*/${FIGURES}/*"))
MP4S=($(find . -name "$GLOB.mp4" -path "./notebooks/*/${FIGURES}/*"))

FILES=("${PDFS[@]}" "${PNGS[@]}" "${MP4S[@]}")

for i in "${FILES[@]}"
    do 
        echo Cleaning up $i 
        rm $i
    done