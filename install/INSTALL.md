## Installation (macOS)

Please note that LUCiFEx is a research code still under active development.

`git clone https://github.com/george-poole/LUCiFEx.git`

`./conda/` contains files to recreate Conda environment. To create a Conda environment named `lucifex`, either do 


* `conda create -n lucifex` followed `conda activate lucifex` and then one of

    * `conda install --file conda_explicit.txt` <br>
    (requirements file created by `conda list --explicit > conda_explicit.txt`)

    * `conda install --file conda.txt` <br>
    (requirements file created by `conda list > conda.txt`)

or do

* `conda env create --name lucifex -f conda_from_history.yml` <br>
(environment file created by `conda env export --from-history > conda_from_history.yml`)

* `conda env create --name lucifex -f conda.yml` <br>
(environment file created by `conda env export > conda.yml`)

Finally `pip install .` (or `pip install -e .` for editable mode).

To build the book of examples and test that everything is working, do `cd book` and `bash build_all.sh`. Errors will be logged in `build_all.log`.

To prevent unnecessary diffs from the notebooks, ensure that `git` has been configured with `nbdime` and `nbstripout` by doing `nbdime config-git --enable` and `nbstripout --install --keep-output`.