@echo off
REM This batch file is equivalent to your Makefile for Windows

REM Check if setup.sh is executable
IF NOT EXIST setup_project.sh (
    echo setup.sh not found
    goto end
)

REM Run setup.sh using Bash
REM This requires Git Bash or a similar bash environment for Windows
bash setup_project.sh

:end
