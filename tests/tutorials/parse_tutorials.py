import os
import argparse

# CELLS that contain code snippets that have to be tested.
# Cell indexing startes from 0.

tutorial_files = ["tutorials/tutorial1.md",
                  "tutorials/tutorial2.md"]

INDICES = {"tutorial1": [1,2,3,4,5,6,8,10,11,12],
           "tutorial2": list(range(1,11))}
def parse_args():
    """Parse command line arguments.

    Return:
        parsed argument object.

    """
    parser = argparse.ArgumentParser(description='Parses tutorials and \
             creates a shell script to test code snippets inside tutorials')

    parser.add_argument('--root_dir', type=str,
                        help='Root dir of AtacWorks',
                        required=True)
    args = parser.parse_args()
    return args


def main():
    """Parse the Markdown files and extract code snippets to test"""
    args = parse_args()

    for infile in tutorial_files:
        begin = 0
        end = 0
        code_to_test = {}
        with open(os.path.join(args.root_dir, infile), "r") as inf:
            code = []
            count = 0
            tutorial = inf.readlines()
            for line in tutorial:
                count = count+1
                if line.find("```") != -1:
                    if begin == 0:
                        begin=1
                    elif begin == 1:
                        begin = 0
                        end = 1
                if begin == 1:
                    code.append(line)
                if end == 1:
                    code.append(line)
                    key = "code" + str(count)
                    code_to_test[key] = code
                    code = []
                    end = 0

        md_file = os.path.basename(infile).split(".")[0]
        out_shell_script = md_file + ".sh"
        with open(out_shell_script, "w") as outfile:
            export_str = "atacworks=" + args.root_dir + "\n"
            outfile.write(export_str)
            for index, (key, value) in enumerate(code_to_test.items()):
                if index not in INDICES[md_file]:
                    continue
                code = value[1:-1]
                code_str = ''.join(string for string in code)
                outfile.write(code_str)

if __name__ == "__main__":
    main()
