
import csv


def save_state_to_csv(state, total_halt, std_dev_halt, beta, reward, filename):
    """
    Save the state (halting numbers and phases) to a CSV file.
    "movingfromNorth", "movingfromEast", "movingfromSouth", "movingfromWest",
    "total_moving", "gamma",
     [total_moving] + [gamma] +
     [state[4]]+[state[5]]+[state[6]]+[state[7]]+
    Parameters:
    - state: The state vector (a combination of normalized halting numbers and traffic light phases).
    - filename: The name of the CSV file to save the state.
    """
    # Open the file in append mode ('a') to add new rows without overwriting existing data
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Check if the file is empty to write the header only once
        if file.tell() == 0:
            writer.writerow(["haltfromNorth", "haltfromEast", "haltfromSouth", "haltfromWest",
                             "total_halt",  "std_dev_halt", "beta", "reward"])  # Writing headers

        # Writing the state as a row in the CSV
        writer.writerow([state[0]]+[state[1]]+[state[2]]+[state[3]]+[total_halt] + [std_dev_halt] + [beta] + [reward])