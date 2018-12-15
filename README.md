# cs221-project

This project focuses on using transfer learning to improve training efficiency for several games contained in the PyGame Learning Environment.

Work was done for Stanford's CS221 class.

Contact: `{cedrick, archow, clomeli}@cs.stanford.edu`

# What's Included

We have a fork of the PyGame-Learning-Environment, where we've inserted our own feature extractor into the PixelCopter game. 
For the sake of brevity, we include the modified file in the `src/pixelcopter/pixelcopter.py`.

Our `src` folder contains several folder for the games that we tried, each with a similar structure.
Beginning training on a game requires installing the packages listed in the `requirements.txt` file in the home directory, then running `python <which-agent-to-train>.py` in one of the `src` folders.
Inside each of these `.py` files is the infrastructure used for training agents on different games.
They mostly follow from the reference implementations in the keras-rl library, with deviations explained in the source file.

Our workflow is generally as follows:
1. run a `.py` file.
2. in a separate terminal, open up an instance of tensorboard so that we may follow its performance
3. determine the best way to change the model and improve performance
4. repeat

We also have a small `analysis` folder, where we kept some of the scripts we used to analyze our results, some of which are stored in `testruns` but are otherwise kept in the `writeup` folder, which contains all the `.tex` files that were used to create our reports. 
