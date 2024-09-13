# MSCI_446_Project

This Repository is dedicated to the development and completion of MSCI 446's (Intro to Machine Learning) course project.

- Dataset
- Quick setup
- Git commands
- Running ETL Script
- Preprocessing
- Models
- Predictions

## Dataset
The dataset was created by us and can be found uploaded to Kaggle [here](https://www.kaggle.com/datasets/jeevanparmar03/nodal-price-prediction-msci-446).

## How do I get set up?

### Summary of set up

Setup is fairly simple, follow the below steps to download all dependencies and to configure anything needed onto your local machine

### Configuration

-`.env` Configuration

### Installing Dependencies and Virtual Environment

#### Manual Approach:

1. Install Python and pip (if you have not done so)
2. `python -m venv venv` or `python3 -m venv venv` (creates virtual environment, this is done one time not every time)
3. Mac: `source venv/bin/activate` (activates virtual environment)
4. Windows: `.\venv\Scripts\activate` (activates virtual environment)

**Make sure you have activated venv before doing any of the following steps or working on the project**

4. `pip install -r requirements.txt`(installs dependencies from the requirements file to the venv)
5. `pip freeze` (outputs list of all dependencies in the venv)
6. `pip freeze > requirements.txt` (overrides old requirements.txt with new dependencies)
7. If for what ever reason you have to deactivate the venv run: `deactivate`

**Before doing anything within the project, make sure you have activated the venv folder**

#### Automatic-ish Approach:

**For Windows:** After cloning, Windows users should run the `setup.bat` file to configure the environment:
Right-click on `setup.bat` and select "Run as administrator" or in the terminal, type:
`.\setup.bat` and press enter.

**Install Xcode if you haven't already (Mac users)**: `xcode-select --install` in the terminal if you want to use this approach

**For Linux/Unix (Mac):** After cloning, Mac/Linux users should open a terminal in the project directory and run:
`make setup`

**Either way make sure the venv file in activated**

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

# Running ETL Script With Driver

The ETL script can be interacted with via the command line to perform various operations such as listing all available sub-folders within the specified root directory or processing data within a specific folder or across all folders. Below are the ways you can run the script depending on your requirements:

## Listing All Sub-folders

To list all available sub-folders in the root directory, use the `--list` option:
`python ETL/src/etl.py --list`
This command will output all sub-folders found in the root directory, for example:
`Available sub-folders:
Gen_Outages
Historical_DA_Prices
Load_Forecast
Solar_Forecast
Wind_Forecast`

## Processing a Specific Folder

If you wish to process CSV files within a specific folder, use the `-f` or `--folder` option followed by the name of the folder:
`python ETL/src/etl.py --folder <Folder_Name>`
For example:
`python ETL/src/etl.py --folder Load_Forecast`
This will execute the ETL process on CSV files located within the `Load_Forecast` subfolder, extracting data, transforming it, and loading it into the MongoDB collection named after the folder.

## Processing All Folders

To process all folders within the root directory, simply run the script without any options:
`python ETL/src/etl.py --all`
This triggers the script to process all sub-folders and their CSV files by default. Ensure the script's default behavior is set to process all folders if no specific folder is provided or no options are used.

## Clearing All Collections in the DB

To remove all documents from every collection in the MongoDB database without deleting the collections themselves, you can use the `--clear` option:

`python ETL/src/etl.py --clear`

This command will clear all documents from each collection within your MongoDB database. It's a powerful operation that makes your collections empty, so use it with caution. The script will output the name of each collection being cleared and confirm once all collections have been processed.

For example, the output might look like this:

`Clearing collection: Gen_Outages
Clearing collection: Historical_DA_Prices
Clearing collection: Load_Forecast
Clearing collection: Solar_Forecast
Clearing collection: Wind_Forecast
All collections have been cleared.`

**Important Note:** This operation will remove all existing data in the collections. Ensure you have backups or do not need the data before executing this command.

## Listing All Collections in the Database

To list all available collections within the MongoDB database, you can use the `--list-collections` option:
`python ETL/src/etl.py --list-collections`
This command will output all collections available in the database, allowing you to see the collections that you can clear or process individually.

## Clearing a Specific Collection

If you want to clear all documents from a specific collection within your MongoDB database, you can use the `--clear-collection` option followed by the name of the collection:

`python ETL/src/etl.py --clear-collection <Collection_Name>`

Replace `<Collection_Name>` with the actual name of the collection you wish to clear. This operation will remove all documents from the specified collection but will not delete the collection itself. Use this command with caution to avoid unintentional data loss.

For example, to clear the Load_Forecast collection:
`python ETL/src/etl.py --clear-collection Load_Forecast`

This will clear all documents from the `Load_Forecast` collection and output a confirmation message:
`Clearing collection: Load_Forecast
Collection Load_Forecast has been cleared.`

**Important Note:** Ensure you have backups or do not need the data before executing this command to avoid unintended data loss.

## Preprocessing

The preprocessing folder holds the notebook used to merge and normalize all datasets together into one dataset

## Models
Holds Neural Networks

## Predictions

Holds notebooks for each type of ML model used (regressions, trees, and a neural network), it's train and test and scores.
