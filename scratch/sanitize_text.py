import os

input_text_path = 'save/unconstrained_vanilla_v8_MP_step50_bs32_0731_split_l1000_test/texts.txt'
test_text_path = 'assets/1000_fuzzy_test.txt'

# Initialize sets
train_set = set()
test_set = set()

# Flag to indicate whether we're reading the test set portion
reading_test_set = False

# Read the input file
with open(input_text_path, 'r') as file:
    for line in file:
        stripped_line = line.strip()  # Remove leading/trailing whitespaces
        
        if not stripped_line:  # If the line is empty (contains only '\n')
            reading_test_set = True
            continue
        
        if reading_test_set:
            if stripped_line not in train_set:
                test_set.add(stripped_line)
        else:
            train_set.add(stripped_line)

# Write the test set to the output file
with open(test_text_path, 'w') as file:
    for item in test_set:
        file.write(f"{item}\n")