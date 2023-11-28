import os

folder_path = "./outputs/"  # Replace with the path to your folder

def check_unknown(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if 'unknown' in line.lower():
                return True
    return False

def main():
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            if check_unknown(file_path):
                print(f"File with 'unknown' content: {filename}")

if __name__ == "__main__":
    main()
