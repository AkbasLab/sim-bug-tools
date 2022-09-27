import sim_bug_tools.utils as utils
# import sim_bug_tools.graphics as graphics

import matplotlib.pyplot as plt
import numpy as np

def main():
    timing = np.array(utils.load("stats/timing.pkl"))
    

    
    
    n = 6.694e97
    avg = timing.mean()
    seconds = n*avg
    print("seconds:", seconds)

    hours = seconds/3600
    print("hours:", hours)

    days = hours/24
    print("days:", days)

    years = days/365
    print(years)



    # Graph
    plt.figure(figsize = (6,4))
    ax = plt.axes()

    ax.set_title("Runtime of 1000 tests.")
    ax.hist(timing, bins=100, color="black")
    
    ax.set_xlabel("Runtime (s)")

    # ax.axvline(x=36, color='b', label='axvline - full height')

    plt.savefig("figs/runtime.png")
    
    return

if __name__ == "__main__":
    main()
