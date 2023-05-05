# How to release a new version of DeepView.Predict
- Go to Github repo and run the action `CI`. You will be prompted to specify the version number.

- This runs a GitHub Action that will take the following steps:
   1. Fetches the repo and its dependencies
   2. Creates a release branch
   3. Updates the version number to the user-specified version using `incremental`
   4. Commits the changes and tag the commit with the version number
   5. Launches an EC2 instance with a T4
   6. The EC2 instance builds the Python build artifacts for Python versions 3.7-3.10
   7. Publishes a release to Github
   8. Create a PR to merge back into main
   9. Publishes to Test PyPI
   10. Publishes to PyPI
   
- This GitHub action is defined under `.github/workflows/whl-build-ec2.yaml`

- This release process follows the release process outlined in [OneFlow](https://www.endoflineblog.com/oneflow-a-git-branching-model-and-workflow).