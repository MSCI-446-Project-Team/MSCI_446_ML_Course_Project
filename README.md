# MSCI_446_Project

This Repository is dedicated to the development and completion of MSCI 446's (Intro to Machine Learning) course project.

- Quick setup
- Git commands
- [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

## How do I get set up?

### Summary of setup

Setup is fairly simple, follow the below steps to download all dependencies and to configure anything needed onto your local machine

### Configuration

-`.env` Configuration

### Installing Dependencies and Virtual Environment

1. Install Python and pip (if you have not done so)
2. `python -m venv venv` or `python3 -m venv venv` (creates virtual environment, this is done one time not every time)
3. `source venv/bin/activate` (activates virtual environment)

**Make sure you have activated venv before doing any of the following steps or working on the project**

4. `pip install -r requirements.txt`(installs dependencies from the requirements file to the venv)
5.
6. `pip freeze` (outputs list of all dependencies in the venv)
7. `pip freeze > requirements.txt` (overrides old requirements.txt with new dependencies)
8. If for what ever reason you have to deactivate the venv run: `deactivate`

**Before doing anything within the project, make sure you have activated the venv folder**

### `.env` Configuration

This project requires certain environment variables to function correctly. Use the provided `.env.template` file as a reference:

1. Copy the `.env.template` file and rename it to `.env`.
2. Fill in the required values as indicated in the template.

## Contribution guidelines

- Writing tests
- Code review
- Other guidelines

## Who do I talk to?

- Repo owner or admin
- Other community or team contact

### Git Commands

All the commands are run in the terminal

## Creating a New Branch

`git checkout -b "your-branch-name"`

## Staging and Committing Changes

```
git add .
git commit -m "Your meaningful commit message here"
```

## Pushing Changes

```
git push -u origin "your-branch-name"  #publishes branch (this is optional but should do)
git push origin "your-branch-name"     #publishes changes

```

## After Pushing Changes to GitHub

1. Go to the pull request tab on the left side of the UI and press create pull request.
2. Make sure you are merging from and to the correct branches
3. Create pull request
4. Approve the pull request
5. Merge the 2 branches
6. Optional: Before merging the 2 branches there is a check box to delete the branch, you can if you want NEVER DELETE MASTER BRANCH
7. If you deleted the branch: go to your local machine and switch from the branch you deleted in the remote repository to the master branch
8. Delete the branch locally (command provided below)

## Switching Between Branches

`git checkout "branch-name"`

## Or to create a new branch and switch to it:

`git checkout -b "new-branch-name"`

## Pulling Changes

```
git checkout "master"            # switch to the main branch
git pull origin "master"         # pull the latest changes
```

## Deleting a Branch

Before deleting a branch first switch to the master branch:
`git checkout "master" `
Then delete the branch, either from the Source control tab or from the terminal:
`git branch -D "branch_name"`
If you want to delete a branch from the remote repository:
`git push --delete "remote_name branch_name"`
