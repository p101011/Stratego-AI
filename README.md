# Stratego-AI

Group project for CSCI 4511w AI course. Group members are Byron Ambright and Peter Day

This Stratego is the 10v10 version (8x8 board) and runs on the command line. All game mechanics are 
contained in main.py. AI.py contains logic for randomly selecting a move. pieces.py holds the classes for
each piece in the game. player.py is used to get a move from each player during play. AITwo.py contains the
logic for intelligently selecting a move. Our AI uses mini-max with alpha-beta pruning and random
configurations of the board (which we called 'musical chairs') to select the best move.

## Usage

cd Stratego\ AI/
python main.py <Red Team AI> <Blue Team AI> <Runs>

## Adding AI's

To add an AI:
1. Name the file to be exactly what the class name is
2. Add file to be within Ai Folder
3. Add Class name to __all__ list in __init__.py