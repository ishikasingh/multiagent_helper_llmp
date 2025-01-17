import re

def convert_barman_to_enabled(input_file, output_file, num_shakers=3):
    # Read the input file
    with open(input_file, 'r') as f:
        content = f.read()

    # Convert PDDL content
    if input_file.endswith('.pddl'):
        # Replace shaker1 declaration with n shakers
        shaker_list = ' '.join(f'shaker{i}' for i in range(1, num_shakers + 1))
        content = re.sub(
            r'(\s+)shaker1 - shaker',
            f'\\1{shaker_list} - shaker',
            content
        )

        # Find all shaker1-related predicates and duplicate them for all shakers
        predicates = [
            r'(\s+)\(ontable shaker1\)',
            r'(\s+)\(clean shaker1\)',
            r'(\s+)\(empty shaker1\)',
            r'(\s+)\(shaker-empty-level shaker1 l0\)',
            r'(\s+)\(shaker-level shaker1 l0\)'
        ]

        for pred in predicates:
            match = re.search(pred, content)
            if match:
                indent = match.group(1)  # Capture the exact indentation
                base_pred = pred.split('shaker1')[1].rstrip('\\)')  # Extract the predicate pattern
                
                # Create the replacement string for all shakers
                replacements = []
                for i in range(1, num_shakers + 1):
                    if 'ontable' in pred:
                        replacements.append(f"{indent}(ontable shaker{i})")
                    elif 'clean' in pred:
                        replacements.append(f"{indent}(clean shaker{i})")
                    elif 'empty' in pred and 'shaker-empty-level' not in pred:
                        replacements.append(f"{indent}(empty shaker{i})")
                    elif 'shaker-empty-level' in pred:
                        replacements.append(f"{indent}(shaker-empty-level shaker{i} l0)")
                    else:  # shaker-level
                        replacements.append(f"{indent}(shaker-level shaker{i} l0)")
                
                replacement = ''.join(replacements)
                content = re.sub(pred, replacement, content)

        # Fix any potential double closing parentheses in the goal section
        content = re.sub(r'\)\)\)', r')))', content)

    # Convert NL content
    else:
        content = content.replace(
            "You have 1 shaker with",
            f"You have {num_shakers} shakers with"
        )
        content = content.replace(
            " levels,",
            " levels each,"
        )

    # Write the output file
    with open(output_file, 'w') as f:
        f.write(content)

# Example usage:
# convert_barman_to_enabled('domains/barman/p01.pddl', 'domains/barman-enabled/p01.pddl', num_shakers=5)
# convert_barman_to_enabled('domains/barman/p01.nl', 'domains/barman-enabled/p01.nl', num_shakers=5)
if __name__ == "__main__":
    for i in range(1, 21):
        num = f"0{i}" if i < 10 else str(i)
        convert_barman_to_enabled(f'domains/barman/p{num}.pddl', f'domains/barman-enabled-2/p{num}.pddl', num_shakers=2)
        convert_barman_to_enabled(f'domains/barman/p{num}.nl', f'domains/barman-enabled-2/p{num}.nl', num_shakers=2)
