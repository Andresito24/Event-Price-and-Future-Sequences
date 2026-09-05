import random

num_samples = 500

with open("sequences.csv", "w") as file:

    # Header
    file.write("x1,x2,x3,x4,x5,label\n")

    for _ in range(num_samples):

        # Generate a sequence of 5 numbers
        sequence = [random.randint(0, 1) for _ in range(10)]

        # Label: 1 if there are at least 3 ones
        label = 1 if sum(sequence) >= 5 else 0

        # Convert sequence to CSV format
        row = ",".join(map(str, sequence))

        file.write(f"{row},{label}\n")