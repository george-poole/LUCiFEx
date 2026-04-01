# Building the Jupyter Book

To execute only notebooks with name matching the glob pattern `XYZ*` do

`bash build.sh "XYZ*"`


For a dry-run do

`bash build.sh --dry "XYZ*"`


For a complete rebuild do

`bash build.sh "*" "" "--allow-errors" "--all"` 