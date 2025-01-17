import re
import sys

def update_pddl_file(file_path, num_robots):
    with open(file_path, 'r') as f:
        content = f.read()

    # Add robot objects
    robot_objects = ' '.join([f'robot{i+1}' for i in range(num_robots)])
    objects_pattern = r'(\(\:objects\s+[^)]+)'
    content = re.sub(objects_pattern, f'\\1 {robot_objects} - robot', content)

    # Find all tools and objects that could be held
    tools_pattern = r'-\s*tool\s+([^-\)]+)'
    wheels_pattern = r'-\s*wheel\s+([^-\)]+)'
    nuts_pattern = r'-\s*nut\s+([^-\)]+)'
    
    tools_match = re.search(tools_pattern, content)
    wheels_match = re.search(wheels_pattern, content)
    nuts_match = re.search(nuts_pattern, content)
    
    holdable_objects = []
    if tools_match:
        holdable_objects.extend(tools_match.group(1).strip().split())
    if wheels_match:
        holdable_objects.extend(wheels_match.group(1).strip().split())
    if nuts_match:
        holdable_objects.extend(nuts_match.group(1).strip().split())

    # Add robot initial states
    robot_states = []
    # Add position states
    robot_states.extend([f'(robot-at robot{i+1} the-hub1)' for i in range(num_robots)])
    # Add holding states
    for obj in holdable_objects:
        robot_states.append(f'(not (exists (?r - robot) (holding ?r {obj})))')

    robot_inits = '\n    '.join(robot_states)
    init_pattern = r'(\(\:init\s+[^)]+)'
    content = re.sub(init_pattern, f'\\1\n    {robot_inits}', content)

    # Write the modified content back
    with open(file_path, 'w') as f:
        f.write(content)

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <number_of_robots>")
        return

    num_robots = int(sys.argv[1])
    if num_robots < 2:
        print("Number of robots must be at least 2")
        return

    # Update p01.pddl through p03.pddl
    for i in range(1, 4):
        file_path = f'p0{i}.pddl'
        try:
            update_pddl_file(file_path, num_robots)
            print(f"Successfully updated {file_path}")
        except Exception as e:
            print(f"Error updating {file_path}: {str(e)}")

if __name__ == "__main__":
    main()
