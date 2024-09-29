README.md

https://neoneye.github.io/arc/?dataset=ARC

Solution approaches that have led to the current state-of-the-art.
As per ARC official website:
These 5 approaches have led to the current state of the art (Taken from https://arcprize.org/guide) :

1. Discrete program search
This was the first domain of solutions that started working well in the original ARCathon competition in 2020 hosted by Lab42. It involves searching through a massive program space in a discrete, step-by-step manner.

2. Ensemble Solutions
This approach consists of piecing together existing publicly available solutions to correctly answer more tasks than any solution achieved alone. This is the approach that was used to get to the current high score.

One thing to consider in utilizing this approach: it's unlikely that an ensemble approach will be able to generalize to correctly solve tasks outside of the public datasets. If you've got your eyes on the Grand Prize, you'll want to create new and novel techniques.

3. Direct LLM Prompting
In this method, contestants use a traditional LLM (like GPT-4) and rely on prompting techniques to solve ARC-AGI tasks. This was found to perform poorly, scoring <5%. Fine-tuning a state-of-the-art (SOTA) LLM with millions of synthetic ARC-AGI examples scores ~10%.

"LLMs like Gemini or ChatGPT [don't work] because they're basically frozen at inference time. They're not actually learning anything." - François Chollet

Additionally, keep in mind that submissions to Kaggle will not have access to the internet. Using a 3rd-party, cloud-hosted LLM is not possible.

See templates for fine-tuning Llama 3b, open source LLM (without fine-tuning it), and using frontier models (Video tutorial, ARC-AGI-Pub only).

4. Domain-Specific Language (DSL) Program Synthesis
This approach involves developing a domain-specific language (DSL). The DSL is designed to encapsulate common concepts such as rotation, mirroring, and other grid transformations that frequently occur in ARC tasks. By defining a set of primitives or basic functions that perform these transformations, solutions can be synthesized by composing these primitives into programs that solve specific tasks.

Program synthesis in this approach involves searching through possible compositions of the DSL primitives to find programs that correctly transform input grids into their corresponding output grids. This search can be brute-force or more sophisticated, but the key idea is to leverage the DSL to build task-specific programs efficiently.

See Michael Hodel's example notebook with this approach.

5. Active inference
More recently, solutions using pre-trained large language models (LLMs) have been attempted. The LLMs are additionally trained on code data, ARC-AGI data, and because there aren’t enough ARC-AGI tasks, you’ll augment this with synthetic ARC-AGI-like data.

The trick to making this LLM based solution work is using active inference. This is the idea that when you’re presented with a test task demonstration examples, fine tune the LLM on those examples. Of course, because there are only a couple of them, you’ll need to expand them artificially to have enough data points to fit your curve.

This unlocks the performance that we see with top solutions. Jack Cole's 34% solution utilizes this approach.

“The fact that this technique has an outsized impact is really interesting” - François Chollet

ARC Prize 2024 Challenge: Vision Model with Reasoning Ability and Intuitive Decision-Making
This repository contains code for developing a model to solve the ARC (Abstraction and Reasoning Corpus) tasks as part of the ARC Prize 2024 competition. The goal is to build a "Vision model with reasoning ability" and an "Intuitive decision-maker" to solve novel tasks using JSON inputs and a 2D graphical interface.

Table of Contents
Project Overview
Installation
Usage
Project Structure
Functions and Modules
Examples
Future Improvements
Contributing
License
Project Overview
The ARC Prize 2024 challenge requires developing an AI model that can:

Understand and interpret visual inputs (grids of colors).
Recognize patterns and transformations from a minimal set of examples.
Generate correct outputs for novel, unseen tasks.
Make intuitive decisions using an ensemble of possible solutions.
The model combines deep learning (for visual understanding) and rule-based reasoning (for pattern recognition and decision-making).

Installation
1. Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/arc-prize-2024.git
cd arc-prize-2024
2. Set Up a Virtual Environment (Optional but Recommended)
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install Required Dependencies
Install all necessary Python libraries using pip:

bash
Copy code
pip install -r requirements.txt
If requirements.txt is not provided, here are some basic libraries to install:

bash
Copy code
pip install numpy pandas matplotlib seaborn scipy json5
Additional libraries might be needed depending on the specific implementations (e.g., torch or tensorflow for deep learning models).

Usage
1. Prepare the Dataset
Ensure you have the following JSON files from the ARC dataset in the data directory:

arc-agi_training-challenges.json
arc-agi_training-solutions.json
arc-agi_evaluation-challenges.json
arc-agi_evaluation-solutions.json
arc-agi_test-challenges.json
sample_submission.json
You can download these files from the competition page on Kaggle or ARCPrize.org and place them in the data folder.

2. Run the Main Code
To execute the main script and start processing the tasks:

bash
Copy code
python arc-mastery.py
3. View Results and Visualizations
The script includes functions to visualize input grids, output predictions, and various intermediate steps. These visualizations will be displayed using matplotlib during the script's execution.

4. Modify or Add New Solvers
To implement additional solvers or modify existing ones, edit the arc-mastery.py file and define new functions under the solvers section.

5. Run Specific Examples
You can run specific task examples by modifying the arc-mastery.py file:

Use the hellow_arc() function to visualize a specific task.
Use the get_arc() function to retrieve task data for testing custom solvers.
Project Structure
bash
Copy code
arc-prize-2024/
│
├── data/                                  # Contains all the dataset files
│   ├── arc-agi_training-challenges.json
│   ├── arc-agi_training-solutions.json
│   ├── arc-agi_evaluation-challenges.json
│   ├── arc-agi_evaluation-solutions.json
│   └── arc-agi_test-challenges.json
│
├── arc-mastery.py                          # Main script containing all functions and solvers
├── requirements.txt                        # List of dependencies for easy installation
└── README.md                               # Documentation for setup and usage
Functions and Modules
Main Functions
load_data(file_path): Loads JSON data from a given file path.
visualize_grid(grid): Visualizes a 2D grid using matplotlib.
extract_features(grid): Extracts features from a grid using a CNN model.
generate_solutions(grid, features): Generates candidate solutions using various transformations.
score_solutions(candidates): Scores each candidate solution based on a heuristic or learned model.
select_best_solution(ranked_candidates): Selects the best solution from the ranked candidates.
process_tasks(tasks): Main loop to process all tasks and visualize results.
Solver Functions
f0000(inp, task): Applies rotations and flips to generate variations of the input grid.
f0001(inp, task): Sorts and draws repetitions of each color.
f0002(inp, task): Concatenates rotated versions to generate complex patterns.
f0003(inp, task): Selects specific zones from the input grid to generate outputs.
Examples
To run specific examples provided in the script:

Example 1: Visualize and solve a task from the training set.
python
Copy code
hellow_arc(185, 'train', True)
Example 4: Generate rotated and flipped versions of an input grid.
python
Copy code
inp = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
clist = f0000(inp, [])
for x in clist:
    plot_pic(x)
Future Improvements
Integrate Deep Learning Models: Replace some heuristic solvers with deep learning models (e.g., Transformers or Graph Neural Networks) for more generalized pattern recognition.
Implement Feedback Loops: Develop feedback mechanisms for continuous learning from mistakes.
Enhance Visualization: Improve the 2D visualization interface with interactivity for better exploration of solutions.
Contributing
We welcome contributions from the community. If you have ideas for new solvers or improvements, feel free to fork the repository and submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Videos relevant for ARC Prize 2024
Refer to ARC prize 2024 Competition:
https://www.kaggle.com/competitions/arc-prize-2024/

I found a number of videos ( freely available online) useful for participation in this competition. I am sharing the links here:

Francois Chollet recommends this method to solve ARC-AGI:
( <3 min video) https://www.youtube.com/watch?v=VqMcfdzqedE

Francois Chollet - LLMs won’t lead to AGI - $1,000,000 Prize to find true solution
( 1 hr, 40 min Video) https://www.youtube.com/watch?v=UakqL6Pj9xo

If an LLM solves this then we'll probably have AGI – Francois Chollet
( 5 min video) https://www.youtube.com/watch?v=wZ0ToxtXz5g

Chollet's ARC Challenge + Current Winners
( 2 hr, 15 mins video) https://www.youtube.com/watch?v=jSAT_RuJ_Cg

Explore ARC-AGI Data + Play
( 11 mins video) https://www.youtube.com/watch?v=LLxiPrIxdqs

Participants in this competition can view these videos and share their findings and comments here.

solved in openai's o1?
I will try to see how far I can solve problems using openai's o1.

Currently, I have confirmed that I can solve the first problem.
I plan to check the rest in the future. I would like to hear your opinions and experimental results.
It is difficult to reflect all the problems.

Thank you for your cooperation.

Here is my result:"007bbfb7"
◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆
User:You are a puzzle solving wizard. You are given a puzzle from the abstraction and reasoning corpus developed by Francois Chollet.<|eot_id|><|start_header_id|>user<|end_header_id|>

Here are the example input and output pairs from which you should learn the underlying rule to later predict the output for the given test input:
[{'input': [[0, 7, 7], [7, 7, 7], [0, 7, 7]], 'output': [[0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7]]}, {'input': [[4, 0, 4], [0, 0, 0], [0, 4, 0]], 'output': [[4, 0, 4, 0, 0, 0, 4, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 0, 4, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0, 0]]}, {'input': [[0, 0, 0], [0, 0, 2], [2, 0, 2]], 'output': [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 2, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 2], [2, 0, 2, 0, 0, 0, 2, 0, 2]]}, {'input': [[6, 6, 0], [6, 0, 0], [0, 6, 6]], 'output': [[6, 6, 0, 6, 6, 0, 0, 0, 0], [6, 0, 0, 6, 0, 0, 0, 0, 0], [0, 6, 6, 0, 6, 6, 0, 0, 0], [6, 6, 0, 0, 0, 0, 0, 0, 0], [6, 0, 0, 0, 0, 0, 0, 0, 0], [0, 6, 6, 0, 0, 0, 0, 0, 0], [0, 0, 0, 6, 6, 0, 6, 6, 0], [0, 0, 0, 6, 0, 0, 6, 0, 0], [0, 0, 0, 0, 6, 6, 0, 6, 6]]}, {'input': [[2, 2, 2], [0, 0, 0], [0, 2, 2]], 'output': [[2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 0, 2, 2, 0, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 2, 0, 2, 2]]}]
Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data.:
[{'input': [[7, 0, 7], [7, 0, 7], [7, 7, 0]], 'output': [[]]}]
What is the output grid? Only provide the output grid in the form as in the example input and output pairs. Do not provide any additional information:<|eot_id|>
Output:

◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆
ChatGPT o1-preview:(actually, the step is in Japanese)
Understanding GPT-3: Using Pattern Recognition

We are looking for a way to find patterns in the input grid to generate the output grid.

Comparing grid sizes

We are looking at the grid sizes of the input and output. The input is 3x3 and the output is 9x9, which is quite different.

Determining the expansion ratio

Since the output is 9x9 and the input is 3x3, we know that there is a rule to expand the input grid to 9x9.

Exploring possible mappings

Looking at the location of the mapping from input to output, we are either trying to put a copy at a specific location, convolve, rotate, reflect, or apply a rule.

Expanding options to consider

What if the output grid self-convolves with the input grid? Or maybe it's superimposed by changing the direction of movement. adicionales_grid_steps**

Determining the location

We are finding the location of the 7 on the input grid and using that location to determine the location of the 7 on the output grid.

Exploring the factors behind the data we are looking at

The 7 appears in multiple locations on the output grid.

Transformations under consideration

The output grid may be formed using alignment of the input grid, convolution with a kernel, or a sum of grids shifted in 8 directions.

I'm starting to see where I need to go.

Identify positions

I'm finding positions with nonzero elements and placing the input grid at the corresponding positions in the output grid. The output grid is 9x9 in size, sized based on the maximum shift.

Alignment, considering overlaps

Trial and error grid alignment

I'm trying to figure out how to place the input grid at each nonzero element position in the 9x9 output grid, or convolve the input grid myself. It looks a bit scattered, though.

Alignment and expansion

I'm initializing the output grid with 9x9 zeros, placing the values ​​of each nonzero position (7) of the input grid, and updating the corresponding region of the output grid.

Looking at the breakdown, it looks like I'm shifting the input grid at each position and updating the output grid. I'm feeling the vibes.

Alignment

I'm creating the output grid by placing a corresponding copy of the grid at each position and shifting it. I'm checking with test inputs to see if this works.

Identifying positions

Finding non-zero positions in the input 3x3 grid and initializing a 9x9 output grid with zeros. Placing the input grid at the non-zero positions.

Scanning and maximizing

As the input grid moves, we search for the maximum value for each position in the output grid. This is how the output grid is formed step by step.

Creating a grid

Creating a 9x9 grid and placing a value of 7. Filling in the first 5 rows and 5 columns and leaving the rest zero.

Checking

The final output grid is 9x9, with some rows and columns filled with 7s. Other rows are completely zero. We are seeing a tendency for excesses and think there may be a specific pattern.

Finding user miscalculation. Checking the input grid positions and finding ways to form the output grid.

Reusing grid structure

Finding ways to reuse the input grid and calculate in 3x3 blocks. Confirming that this results in a 9x9 output grid.

Figuring out matrix type

Calculating the Kuronecker product to create a 9x9 matrix. Each element is calculated by multiplying the rows and columns of A.

Matrix calculation

C is calculated by multiplying each element of A and B. Zeros and 7s are used in some parts.

Looking at the block arrangement

The first three blocks can be calculated by moving down and to the right while calculating each block.

Checking block operations

Setting values ​​to cc[0:3][6:9], scaling A at coordinates (1,0), (1,1), and (1,2). Next, scaling is being performed for coordinates (2,0), (2,1), and (2,2).

Exploring the relationship between position and shift
It is interesting to see that non-zeros act as shifts by overlapping the positions of non-zero elements in the input grid.

Exploring new approaches

I am trying various methods, such as summing the cross products of the rows and columns of the input grid, calculating the "distance" of non-zero elements, and convolution using logical OR.

Exploring ways to form

We are exploring how to overlap input grids, calculate the sum of nonzero elements, and record the nonzero positions.

Exploring and adapting

Because of the time constraint, we are exploring how to accept the output value of each position as nonzero and use the input grid and its inverse to form the output grid.

Exploring

We are exploring how to calculate the output grid based on the positions of nonzero elements in the input grid. Mathematical techniques such as Hanyu multiplication may be used from the positions of nonzero elements at each position.

Options to consider

We are exploring the possibility of the input grid being repeated three times in a 3x3 grid format. As we continue our research focused on environmental assessment of vegetables, our goal is to develop an efficient algorithm.

Unravelling the problem

We can see that the output grid repeats the input grid in 3x3 tiles. There are certain rules for the placement /Users/seungwonleeof the tiles.

Exploring placement of tiles

Placing a 3x3 input grid in a 9x9 output grid. Placing copies at the nonzero element positions of the input grid.

Placing tiles

Pasting copies at the positions of the output grid that correspond to the nonzero positions in the input grid.

The progress of a CHUNK is as follows:

At each position, the input grid values ​​are transferred to the output grid.

The visual arrangement of the grid is tidied up.

json
[[7, 0, 7, 0, 0, 0, 7, 0, 7],
[7, 0, 7, 0, 0, 0, 7, 0, 7],
[7, 7, 0, 0, 0, 0, 7, 7, 0],
[7, 0, 7, 0, 0, 0, 7, 0, 7],
[7, 0, 7, 0, 0, 0, 7, 0, 7],
[7, 7, 0, 0, 0, 0, 7, 7, 0],
[7, 0, 7, 7, 0, 7, 0, 0, 0],
[7, 0, 7, 7, 0, 7, 0, 0, 0],
[7, 7, 0, 7, 7, 0, 0, 0, 0]]

