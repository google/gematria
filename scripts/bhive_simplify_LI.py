import re
import sys

def simplify_live_intervals(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    simplified_outputs = {}
    simplified_output = []
    intervals_section = False
    machine_instrs_section = False
    basic_block_name = None
    basic_block_start = None
    previous_line_end = None
    function_name = None

    for line in lines:
        # Extract function name using regex
        if machine_instrs_section and line.startswith('# Machine code for function'):
            match = re.search(r'function\s+([^\s:]+)', line)
            if match:
                function_name = match.group(1)
                # insert function name to the beginning of simplified_output
                simplified_output.insert(0, f'{function_name}\n')
            continue

        # Filter out comments and irrelevant lines
        if ';' in line or line.strip() == '':
            continue

        if '********** INTERVALS **********' in line:
            simplified_output = []
            if intervals_section and basic_block_name is not None:
                simplified_output.append(f"{basic_block_name}: {basic_block_start}B {previous_line_end}B\n")
            intervals_section = True
            machine_instrs_section = False
            basic_block_name = None
            basic_block_start = None
            previous_line_end = None
            continue

        if 'RegMasks:' in line and intervals_section:
            simplified_output.append(line)
            continue

        if '********** MACHINEINSTRS **********' in line:
            intervals_section = False
            machine_instrs_section = True
            continue

        if '# End machine code for function' in line:
            machine_instrs_section = False
            intervals_section = False
            if basic_block_name is not None:
                simplified_output.append(f"{basic_block_name}: {basic_block_start}B {previous_line_end}B\n")
                basic_block_name = None
            simplified_outputs[function_name] = simplified_output
            continue

        if intervals_section:
            simplified_output.append(line)
        if machine_instrs_section:
            match = re.search(r'bb\.\d+\.(BB_\d+)', line)
            if match:
                if basic_block_name is not None:
                    simplified_output.append(f"{basic_block_name}: {basic_block_start}B {previous_line_end}B\n")
                basic_block_start = line.split()[0].rstrip('B')
                basic_block_name = match.group(1)
            previous_line_end = line.split()[0].rstrip('B') if 'B' in line else None

    if basic_block_name is not None and not machine_instrs_section:
        simplified_output.append(f"{basic_block_name}: {basic_block_start}B {previous_line_end}B\n")

    with open(output_file, 'w') as file:
        for function_name, simplified_output in simplified_outputs.items():
            for line in simplified_output:
                file.write(line)
# Example usage

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python bhive_simplify_LI.py <input_file> <output_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    simplify_live_intervals(input_file, output_file)