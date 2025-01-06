import re

def convert_barman_to_enabled(input_file, output_file):
    # Read the input file
    with open(input_file, 'r') as f:
        content = f.read()

    # Convert PDDL content
    if input_file.endswith('.pddl'):
        # Replace shaker1 declaration with three shakers
        content = re.sub(
            r'(\s+)shaker1 - shaker',
            r'\1shaker1 shaker2 shaker3 - shaker',
            content
        )

        # Find all shaker1-related predicates and duplicate them for shaker2 and shaker3
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
                if 'ontable' in pred:
                    replacement = f"{indent}(ontable shaker1){indent}(ontable shaker2){indent}(ontable shaker3)"
                elif 'clean' in pred:
                    replacement = f"{indent}(clean shaker1){indent}(clean shaker2){indent}(clean shaker3)"
                elif 'empty' in pred and 'shaker-empty-level' not in pred:
                    replacement = f"{indent}(empty shaker1){indent}(empty shaker2){indent}(empty shaker3)"
                elif 'shaker-empty-level' in pred:
                    replacement = f"{indent}(shaker-empty-level shaker1 l0){indent}(shaker-empty-level shaker2 l0){indent}(shaker-empty-level shaker3 l0)"
                else:  # shaker-level
                    replacement = f"{indent}(shaker-level shaker1 l0){indent}(shaker-level shaker2 l0){indent}(shaker-level shaker3 l0)"
                
                content = re.sub(pred, replacement, content)

        # Fix any potential double closing parentheses in the goal section
        content = re.sub(r'\)\)\)', r')))', content)

    # Convert NL content
    else:
        content = content.replace(
            "You have 1 shaker with",
            "You have 3 shakers with"
        )
        content = content.replace(
            " levels,",
            " levels each,"
        )

    # Write the output file
    with open(output_file, 'w') as f:
        f.write(content)

# Example usage:
# convert_barman_to_enabled('domains/barman/p01.pddl', 'domains/barman-enabled/p01.pddl')
# convert_barman_to_enabled('domains/barman/p01.nl', 'domains/barman-enabled/p01.nl')
if __name__ == "__main__":
    for i in range(1, 21):
        num = f"0{i}" if i < 10 else str(i)
        convert_barman_to_enabled(f'domains/barman/p{num}.pddl', f'domains/barman-enabled/p{num}.pddl')
        convert_barman_to_enabled(f'domains/barman/p{num}.nl', f'domains/barman-enabled/p{num}.nl')
