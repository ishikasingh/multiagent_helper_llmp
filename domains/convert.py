import os
import re
from pathlib import Path
import argparse
import shutil

AGENT_PREDICATES = {
    "barman-multi": ['handempty', 'holding'],
    "blocksworld-multi": ['arm-empty', 'holding'],
    "termes-multi": ['has-block', 'at'],
    "tyreworld-multi": [],
    "grippers-multi": ['at-robby', 'free', 'carry'],
}

def reconstruct_pddl(sections):
    """Reconstruct PDDL file from sections."""
    # Start with the header
    result = sections.get('header', '(define (problem prob)') + '\n'
    
    # Add sections in the correct order
    section_order = ['domain', 'objects', 'init', 'goal']
    for section in section_order:
        if section in sections:
            content = sections[section]
            if not content.endswith(')'):
                content += ')'
            result += content + '\n'
    
    # Add final closing parenthesis
    result = result.rstrip() + '\n)'
    return result

def modify_pddl_for_n_robots(content, n_robots, domain_name):
    """Modify PDDL content for n robots."""
    # Split content into init and goal parts
    if '(:goal' in content:
        init_part, goal_part = content.split('(:goal')
    else:
        init_part = content
        goal_part = ''
    
    # Modify robot declarations in init part
    robot_objects = ' '.join([f'robot{i}' for i in range(1, n_robots + 1)])
    init_part = re.sub(r'robot\d+(?:\s+robot\d+)* - robot', f'{robot_objects} - robot', init_part)
    
    # Get domain-specific predicates
    domain_key = f"{domain_name}-multi"
    predicates = AGENT_PREDICATES.get(domain_key, [])
    
    # Special handling for termes domain
    if domain_name == 'termes':
        # Handle initial state predicates
        depot_pos = None
        # Find the depot position
        depot_match = re.search(r'\(IS-DEPOT\s+(pos-\d+-\d+)\)', init_part)
        if depot_match:
            depot_pos = depot_match.group(1)
            
        # Replace all robot at predicates
        at_predicates = '\n    '.join([
            f'(at robot{i} {depot_pos})'
            for i in range(1, n_robots + 1)
        ])
        init_part = re.sub(
            r'\(at robot\d+ [^)]+\)(?:\s*\(at robot\d+ [^)]+\))*',
            at_predicates,
            init_part
        )
        
        # Handle goal state predicates
        if goal_part:
            # Add not-has-block for all robots
            has_block_predicates = '\n    '.join([
                f'(not (has-block robot{i}))'
                for i in range(1, n_robots + 1)
            ])
            # Replace existing not-has-block predicates or add them before the final closing parenthesis
            if '(not (has-block robot' in goal_part:
                goal_part = re.sub(
                    r'\(not \(has-block robot\d+\)\)(?:\s*\(not \(has-block robot\d+\)\))*',
                    has_block_predicates,
                    goal_part
                )
            else:
                goal_part = goal_part.rstrip(')')
                goal_part += f'\n    {has_block_predicates}\n)'
    
    # Modify robot-specific predicates in init part
    for predicate in predicates:
        if predicate == 'handempty':
            # Special handling for barman domain with left/right hands
            robot_hands = '\n  '.join([
                f'(handempty robot{i} left)\n  (handempty robot{i} right)'
                for i in range(1, n_robots + 1)
            ])
            init_part = re.sub(
                r'\(handempty robot\d+ (?:left|right)\)(?:\s*\(handempty robot\d+ (?:left|right)\))*',
                robot_hands,
                init_part
            )
        else:
            # Handle other predicates
            robot_predicates = '\n  '.join([
                f'({predicate} robot{i})'
                for i in range(1, n_robots + 1)
            ])
            init_part = re.sub(
                rf'\({predicate} robot\d+\)(?:\s*\({predicate} robot\d+\))*',
                robot_predicates,
                init_part
            )
    
    # Special handling for grippers domain
    if domain_name == 'grippers':
        # Add gripper objects for each robot
        gripper_pattern = r'(rgripper\d+ lgripper\d+(?:\s+rgripper\d+ lgripper\d+)*) - gripper'
        gripper_objects = ' '.join([f'rgripper{i} lgripper{i}' for i in range(1, n_robots + 1)])
        init_part = re.sub(gripper_pattern, f'{gripper_objects} - gripper', init_part)
        
        # Remove all existing robot-related predicates first
        init_part = re.sub(r'\(free robot\d+ [rl]gripper\d+\)\s*', '', init_part)
        init_part = re.sub(r'\(at-robby robot\d+ room1\)\s*', '', init_part)
        
        # Add predicates once
        new_predicates = []
        
        # Add at-robby predicates for each robot
        for i in range(1, n_robots + 1):
            new_predicates.append(f'(at-robby robot{i} room1)')
        
        # Add free predicates for each robot's grippers
        for i in range(1, n_robots + 1):
            new_predicates.append(f'(free robot{i} rgripper{i})')
            new_predicates.append(f'(free robot{i} lgripper{i})')
        
        # Insert all predicates at the start of init
        init_split = init_part.split('(:init')
        init_part = f'{init_split[0]}(:init\n  ' + '\n  '.join(new_predicates) + init_split[1]
    
    # Reconstruct the file
    if goal_part:
        return init_part + '(:goal' + goal_part
    else:
        return init_part + ')'

def create_n_robot_domain(source_dir, dest_dir, n_robots):
    """Create n-robot version of domain files."""
    domain_name = os.path.basename(source_dir).replace('-multi', '')
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Copy domain.pddl unchanged
    domain_file = Path(source_dir) / 'domain.pddl'
    if domain_file.exists():
        shutil.copy(domain_file, dest_path / 'domain.pddl')
        print(f"Copied {dest_path / 'domain.pddl'}")
    
    # Process problem files
    for file in Path(source_dir).glob('p*.pddl'):
        print(f"Processing {file}")
        content = file.read_text()
        new_content = modify_pddl_for_n_robots(content, n_robots, domain_name)
        
        # Debug output
        print(f"\nGenerated content:\n{new_content}")
        
        dest_file = dest_path / file.name
        dest_file.write_text(new_content)
        print(f"Created {dest_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert multi-robot domain to n-robot domain')
    parser.add_argument('domain', help='Name of the domain to convert (e.g., barman, blocksworld)')
    parser.add_argument('--n-robots', type=int, default=4, help='Number of robots (default: 4)')
    parser.add_argument('--base-dir', default='', help='Base directory containing domain folders (default: domains)')
    
    args = parser.parse_args()
    
    # Construct source and destination paths
    source_dir = Path(args.base_dir) / f"{args.domain}-multi/"
    dest_dir = Path(args.base_dir) / f"{args.domain}-multi-{args.n_robots}"
    
    if not source_dir.exists():
        print(f"Error: Domain '{args.domain}-multi' not found in {args.base_dir}")
        return
    
    print(f"\nProcessing domain: {args.domain}")
    create_n_robot_domain(source_dir, dest_dir, args.n_robots)
    print(f"\nDone! Files created in {dest_dir}")

if __name__ == "__main__":
    main()
