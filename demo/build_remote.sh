GLOB=$1
BUILD=$2

if [ ${#GLOB} -ne 0 ]
then
echo Finding notebooks to execute before building...
bash ./nbconvert.sh $GLOB
fi

jupyter-book build . $BUILD
ghp-import -n -p -f ./_build/html