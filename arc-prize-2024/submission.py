
import json
import numpy as np
import logging

# Set up logging
logging.basicConfig(filename='arc_submission.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(message)s')

# Define paths
train1_path = '/Users/seungwonlee/ARC Prize 2024/ARCPrice2024/arc-prize-2024/arc-agi_training_challenges.json'
train2_path = '/Users/seungwonlee/ARC Prize 2024/ARCPrice2024/arc-prize-2024/arc-agi_training_solutions.json'
eval1_path = '/Users/seungwonlee/ARC Prize 2024/ARCPrice2024/arc-prize-2024/arc-agi_evaluation_challenges.json'
eval2_path = '/Users/seungwonlee/ARC Prize 2024/ARCPrice2024/arc-prize-2024/arc-agi_evaluation_solutions.json'
test_path = '/Users/seungwonlee/ARC Prize 2024/ARCPrice2024/arc-prize-2024/arc-agi_test_challenges.json'
sample_path = '/Users/seungwonlee/ARC Prize 2024/ARCPrice2024/arc-prize-2024/sample_submission.json'


def set_submission():
    logging.info("Starting submission generation...")
    
    try:
        # Open sample submission file
        logging.info("Opening sample submission file...")
        with open(sample_path, 'r') as f:
            sample_sub = json.load(f)
    except FileNotFoundError:
        logging.error(f"Sample submission file not found at {sample_path}.")
        raise
    except json.JSONDecodeError:
        logging.error("Error decoding JSON from the sample submission file.")
        raise

    try:
        # Open test file
        logging.info(f"Opening test file: {test_path}")
        with open(test_path, 'r') as f:
            tasks_name = list(json.load(f).keys())
            tasks_file = list(json.load(f).values())
    except FileNotFoundError:
        logging.error(f"Test file not found at {test_path}.")
        raise
    except json.JSONDecodeError:
        logging.error("Error decoding JSON from the test file.")
        raise

    logging.info("Processing submission data...")

    # Initialize a list to store answers
    answer = []

    for n in range(len(tasks_file)):
        task = tasks_file[n]
        len_train = len(task['train'])

        for sol in solvers:
            counter = np.zeros(len(version))

            # Check if the solutions match the training data
            for e in range(len_train):
                try:
                    inp_e = np.array(task['train'][e]['input'])
                    out_e = np.array(task['train'][e]['output'])
                    clist = solvers[sol](inp_e, task)
                    
                    for v in range(version[sol]):
                        if np.array_equal(clist[v], out_e):
                            counter[v] += 1
                except Exception as ex:
                    logging.error(f"Error in solving task {n} with solver {sol}: {ex}")
                    continue

            # Assigning predictions to the submission file
            for v in range(len(counter)):
                if counter[v] == len_train:
                    try:
                        for i in range(len(task['test'])):
                            inp_test = np.array(task['test'][i]['input'])
                            clist = solvers[sol](inp_test, task)
                            answer_ = clist[v].tolist()

                            sample_sub[tasks_name[n]][i]['attempt_1'] = answer_
                            if answer_ not in answer:
                                answer.append(answer_)
                            logging.info(f"Task {tasks_name[n]}, Item {i}: Solver {sol} - Version {v} assigned to Attempt 1")
                            plot_pic(answer_)
                    except Exception as ex:
                        logging.error(f"Error generating Attempt 1 for Task {tasks_name[n]}: {ex}")
                        continue

                elif counter[v] == len_train - 1:
                    try:
                        for i in range(len(task['test'])):
                            inp_test = np.array(task['test'][i]['input'])
                            clist = solvers[sol](inp_test, task)
                            answer_ = clist[v].tolist()

                            sample_sub[tasks_name[n]][i]['attempt_2'] = answer_
                            if answer_ not in answer:
                                answer.append(answer_)
                            logging.info(f"Task {tasks_name[n]}, Item {i}: Solver {sol} - Version {v} assigned to Attempt 2")
                            plot_pic(answer_)
                    except Exception as ex:
                        logging.error(f"Error generating Attempt 2 for Task {tasks_name[n]}: {ex}")
                        break

    # Save the submission file
    try:
        with open('submission.json', 'w') as file:
            json.dump(sample_sub, file, indent=4)
            logging.info("Submission file 'submission.json' created successfully.")
    except Exception as ex:
        logging.error(f"Error saving submission file: {ex}")
        raise
    
    return sample_sub
    
def save_submission(submission, filename='submission.json'):
    logging.info(f"Saving submission to {filename}...")
    try:
        with open(filename, 'w') as f:
            json.dump(submission, f, indent=4)
        logging.info(f"Submission saved successfully to {filename}")
    except Exception as e:
        logging.error(f"Error saving submission: {e}", exc_info=True)
        raise e

# Example usage
if __name__ == "__main__":
    try:
        submission = set_submission()
        print("Submission completed successfully.")
    except Exception as e:
        print(f"An error occurred during submission: {e}")
        logging.error(f"Submission process failed: {e}", exc_info=True)
