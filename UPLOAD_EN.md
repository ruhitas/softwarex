# GitHub upload package

Upload the contents of the `repository` folder to the root of your GitHub code repository. The README, source code, dependencies, the data split manifest and the saved predictions are in this folder.

The three `.pt` model files in the `release_assets` folder exceed GitHub's upload size limit, so they are shared via Google Drive instead of a GitHub Release, as a single zip archive containing all three files:

https://drive.google.com/file/d/15mHnY6mqvAymrpn6LghK3quhytknkBJX/view?usp=drive_link

Users download and extract the archive, then place the three `.pt` files (`base_swin.pt`, `ce_tail.pt`, `ce_gated.pt`) directly in the repository's `weights` folder. This Google Drive link is already referenced in the README; use both the real GitHub repository address and this Drive link in the manuscript's Code availability section.

Nothing has been uploaded or published to GitHub yet. Raw data, personal computer paths, manuscript Word files, third-party PDFs and the EPPO map were not included in the package. Attribution to the data producers and the data source are in the README.

No reuse license has yet been chosen for the software. The README states this explicitly; no license such as MIT/Apache has been granted on your behalf.

Code comments and the README are in English. The saved results can be verified without the weights. Retraining requires the raw data, CUDA and the provided starting checkpoint. The hyperparameter search for the earlier starting model is outside the scope of this package.
